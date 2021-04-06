import argparse
import os
from datetime import datetime
import tensorflow as tf
import random
import numpy as np
import scipy.io
from scipy.ndimage.filters import gaussian_filter1d
import scipy.special
import pickle
from dataPreprocessing import prepareDataCubesForRNN
import sys

class charSeqRNN(object):
    """
    This class encapsulates all the functionality needed for training, loading and running the handwriting decoder RNN. 
    To use it, initialize this class and then call .train() or .inference(). It can also be run from the command line (see bottom
    of the script). The args dictionary passed during initialization is used to configure all aspects of its behavior.
    """
    def __init__(self, args):
        """
        This function initializes the entire tensorflow graph, including the dataset pipeline and RNN. 
        Along the way, it loads all relevant data and label files needed for training, and initializes the RNN variables to
        default values (or loads them from a specified file). After initialization is complete, we are ready
        to either train (charSeqRNN.train) or infer (charSeqRNN.inference). 
        """
        self.args = args

        #parse whether we are loading a model or not, and whether we are training or 'running' (inferring)
        if self.args['mode']=='train':
            self.isTraining = True
            ckpt = tf.train.get_checkpoint_state(self.args['loadDir'])
            if ckpt==None:
                #Nothing to load (no checkpoint found here), so we won't resume or try to load anything
                self.loadingInitParams = False
                self.resumeTraining = False
            elif self.args['loadDir']==self.args['outputDir']:
                #loading from the same place we are saving - assume we are resuming training
                self.loadingInitParams = True
                self.resumeTraining = True               
            else:
                #otherwise we will load params but not try to resume a training run, we'll start over
                self.loadingInitParams = True
                self.resumeTraining = False   
                
        elif self.args['mode']=='infer':
            self.isTraining = False
            self.loadingInitParams = True
            self.resumeTraining = False
        
        #count how many days of data are specified
        self.nDays = 0
        for t in range(30):
            if 'labelsFile_'+str(t) not in self.args.keys():
                self.nDays = t
                break
        
        #load data, labels, train/test partitions & synthetic .tfrecord files for all days
        neuralCube_all, targets_all, errWeights_all, numBinsPerTrial_all, cvIdx_all, recordFileSet_all = self._loadAllDatasets()
        
        #define the input & output dimensions of the RNN
        nOutputs = targets_all[0].shape[2]
        nInputs = neuralCube_all[0].shape[2]
        
        #this is used later in inference mode
        self.nTrialsInFirstDataset = neuralCube_all[0].shape[0]
            
        #random variable seeding
        if self.args['seed']==-1:
            self.args['seed']=datetime.now().microsecond
        np.random.seed(self.args['seed'])
        tf.set_random_seed(self.args['seed'])
                                        
        #Start tensorflow
        self.sess = tf.Session()
        
        #--------------Dataset pipeline--------------
        #First we put the datasets on the graph. It's a bit tricky since we need to be able to select between different
        #days randomly for each minibatch, so we need to place this selection mechanism on the tensorflow graph. 
        allSynthIterators = []
        allRealIterators = []
        allValIterators = []
        self.daysWithValData = []

        #The following if statement constructs the dataset iterators.
        if self.isTraining:
            #We are in training mode. For each day, we make 'synthetic', 'real', and 'validation' tensorflow dataset iterators.
            for dayIdx in range(self.nDays):
                #--training data stream (synthetic data)--
                if self.args['synthBatchSize']>0:
                    mapFnc = lambda singleExample: parseDataset(singleExample, self.args['timeSteps'], nInputs, nOutputs,
                                                                whiteNoiseSD=self.args['whiteNoiseSD'],
                                                                constantOffsetSD=self.args['constantOffsetSD'],
                                                                randomWalkSD=self.args['randomWalkSD']) 

                    newDataset = tf.data.TFRecordDataset(recordFileSet_all[dayIdx])
                    newDataset = newDataset.apply(tf.data.experimental.map_and_batch(map_func=mapFnc, 
                                                                                     batch_size=self.args['synthBatchSize'], 
                                                                                     drop_remainder=True))
                    newDataset = newDataset.apply(tf.data.experimental.shuffle_and_repeat(int(4)))
                    synthIter = newDataset.make_one_shot_iterator()
                else:
                    synthIter = []

                #--training data stream (real data, train partition)--
                realDataSize = self.args['batchSize'] - self.args['synthBatchSize']
                trainIdx = cvIdx_all[dayIdx]['trainIdx']
                valIdx = cvIdx_all[dayIdx]['testIdx']
                
                if realDataSize>0:
                    realIter = self._makeTrainingDatasetFromRealData(neuralCube_all[dayIdx][trainIdx,:,:], 
                                                                    targets_all[dayIdx][trainIdx,:,:], 
                                                                    errWeights_all[dayIdx][trainIdx,:], 
                                                                    numBinsPerTrial_all[dayIdx][trainIdx,np.newaxis],
                                                                    realDataSize,
                                                                    addNoise=True)
                else:
                    realIter = []
                
                #--validation data stream (real data, test partition)--
                if len(valIdx)==0:
                    valIter = realIter
                    valDataExists = False
                else:
                    valIter = self._makeTrainingDatasetFromRealData(neuralCube_all[dayIdx][valIdx,:,:], 
                                                targets_all[dayIdx][valIdx,:,:], 
                                                errWeights_all[dayIdx][valIdx,:], 
                                                numBinsPerTrial_all[dayIdx][valIdx,np.newaxis],
                                                self.args['batchSize'],
                                                addNoise=False)
                    valDataExists = True
                    
                allSynthIterators.append(synthIter)
                allRealIterators.append(realIter)
                allValIterators.append(valIter)
                if valDataExists:
                    self.daysWithValData.append(dayIdx)
        else:
            #We are in inference mode. We make a tensorflow iterator for the real data stream only 
            #(no synthetic data or validation data). Also, we place only the first days' data on the graph.
            #Inference mode currently only supports running through a single dataset at a time.
            newDataset = tf.data.Dataset.from_tensor_slices((neuralCube_all[0].astype(np.float32), 
                                                             targets_all[0].astype(np.float32), 
                                                             errWeights_all[0].astype(np.float32),
                                                             numBinsPerTrial_all[0].astype(np.int32)))
            newDataset = newDataset.repeat()
            newDataset = newDataset.batch(self.args['batchSize'])

            iterator = newDataset.make_initializable_iterator()
            self.sess.run(iterator.initializer)

            allRealIterators.append(iterator)
            allSynthIterators.append([])
            allValIterators.append(iterator)
        
        #With the dataset iterators in hand, we now construct the selection mechanism to switch between different
        #days for each minibatch. As part of this, we also have to combine the real data and synthetic data into a single minibatch.
        #Note that 'dayNum' selects between the days of data, while 'datasetNum' also selects between train vs. test datasets.
        #Even datasetNums are training datasets and odd datasetNums are validation datasets. 
        self.datasetNumPH = tf.placeholder(tf.int32, shape=[])
        self.dayNumPH = tf.placeholder(tf.int32, shape=[])

        def pruneValDataset(valIter):
            inp, targ, weight, bins = valIter.get_next()
            return inp, targ, weight
            
        def makeValDatasetFunc(x):
            return lambda: pruneValDataset(allValIterators[x])
        
        def combineSynthAndReal(synthIter, realIter):
            if synthIter==[]:
                inp, targ, weight, bins = realIter.get_next()
            elif realIter==[]:
                inp, targ, weight = synthIter.get_next()
            else:
                inp_r, targ_r, weight_r, bins_r = realIter.get_next()
                inp_s, targ_s, weight_s = synthIter.get_next()
                
                inp = tf.concat([inp_s, inp_r],axis=0)
                targ = tf.concat([targ_s, targ_r],axis=0)
                weight = tf.concat([weight_s, weight_r],axis=0)
            
            return inp, targ, weight
        
        def makeTrainDatasetFunc(x):
            return lambda: combineSynthAndReal(allSynthIterators[x], allRealIterators[x])
        
        branchFuns = []
        for datIdx in range(self.nDays):
            branchFuns.extend([makeTrainDatasetFunc(datIdx), makeValDatasetFunc(datIdx)])

        #These variables ('batchInputs', 'batchTargets', 'batchWeight') are the output of the day-selector mechanism
        #and are all that is needed moving forward to define the RNN cost function.
        self.batchInputs, self.batchTargets, self.batchWeight = tf.switch_case(self.datasetNumPH, branchFuns)
    
        self.batchWeight.set_shape([self.args['batchSize'], self.args['timeSteps']])
        self.batchInputs.set_shape([self.args['batchSize'], self.args['timeSteps'], nInputs])
        self.batchTargets.set_shape([self.args['batchSize'], self.args['timeSteps'], nOutputs])

        #--------------RNN Graph--------------
        #First, some simple Gaussian smoothing.     
        if self.args['smoothInputs']==1:
            self.batchInputs = gaussSmooth(self.batchInputs, kernelSD=4/self.args['rnnBinSize'])
            
        #Define the RNN start state, which is trainable.
        if self.args['directionality']=='bidirectional':
            biDir = 2
        else:
            biDir = 1
            
        self.rnnStartState = tf.get_variable('RNN_layer0/startState', [biDir, 1, self.args['nUnits']], dtype=tf.float32, initializer=tf.zeros_initializer, trainable=bool(self.args['trainableBackEnd']))
        
        #tile the state across all elements of the minibatch
        initRNNState = tf.tile(self.rnnStartState, [1, self.args['batchSize'], 1])

        #Define the day-specific input layers.
        self.dayToLayerMap = eval(self.args['dayToLayerMap'])
        self.dayProbability = eval(self.args['dayProbability'])
        self.nInpLayers = len(np.unique(self.dayToLayerMap))
        
        self.inputFactors_W_all = []
        self.inputFactors_b_all = []
        for inpLayerIdx in range(self.nInpLayers):
            self.inputFactors_W_all.append(tf.get_variable("inputFactors_W_"+str(inpLayerIdx),
                                             initializer=np.identity(nInputs).astype(np.float32),
                                             trainable=bool(self.args['trainableInput'])))

            self.inputFactors_b_all.append(tf.get_variable("inputFactors_b_"+str(inpLayerIdx),
                                             initializer=np.zeros([nInputs]).astype(np.float32),
                                             trainable=bool(self.args['trainableInput'])))
            
        #Define the selector mechanism that chooses which input layer to use depending on which day we have selected
        #for the minibatch.
        def makeFactorsFunc(x):
            return lambda: (self.inputFactors_W_all[self.dayToLayerMap[x]], self.inputFactors_b_all[self.dayToLayerMap[x]])
        
        branchFuns_inpLayers = []
        for dayIdx in range(self.nDays):
            branchFuns_inpLayers.append(makeFactorsFunc(dayIdx))
        
        #inp_W and inp_b are the chosen input layer variables
        inp_W, inp_b = tf.switch_case(self.dayNumPH, branchFuns_inpLayers)
        
        #'inputFactors' are the transformed inputs which should now be in a common space across days.
        self.inputFactors = tf.matmul(self.batchInputs, tf.tile(tf.expand_dims(inp_W,0), [self.args['batchSize'], 1, 1])) + inp_b
        
        #Now define the two GRU layers. Layer 1, which runs at a high frequency:
        self.rnnOutput, self.rnnWeightVars = cudnnGraphSingleLayer(self.args['nUnits'], 
                                                                   initRNNState, 
                                                                   self.inputFactors, 
                                                                   self.args['timeSteps'], 
                                                                   self.args['batchSize'], 
                                                                   nInputs, 
                                                                   self.args['directionality'])

        #Layer 2, which runs at a slower frequency (defined by 'skipLen'):
        nSkipInputs = self.args['nUnits']
        skipLen = self.args['skipLen']
        
        with tf.variable_scope("layer2"):
            self.rnnOutput2, self.rnnWeightVars2 = cudnnGraphSingleLayer(self.args['nUnits'], 
                                                                         initRNNState, 
                                                                         self.rnnOutput[:,0::skipLen,:], 
                                                                         self.args['timeSteps']/skipLen, 
                                                                         self.args['batchSize'], 
                                                                         self.args['nUnits']*biDir,  
                                                                         self.args['directionality'])
            
        #Finally, define the linear readout layer.
        self.readout_W = tf.get_variable("readout_W",
                            shape=[biDir*self.args['nUnits'], nOutputs],
                            initializer=tf.random_normal_initializer(dtype=tf.float32, stddev=0.05),
                            trainable=bool(self.args['trainableBackEnd']))

        self.readout_b = tf.get_variable("readout_b",
                                    shape=[nOutputs],
                                    initializer=tf.zeros_initializer(dtype=tf.float32),
                                    trainable=bool(self.args['trainableBackEnd']))

        tiledReadoutWeights = tf.tile(tf.expand_dims(self.readout_W,0), [self.args['batchSize'], 1, 1])
        self.logitOutput_downsample = tf.matmul(self.rnnOutput2, tiledReadoutWeights) + self.readout_b
        
        #Up-sample the outputs to the original time-resolution (needed b/c layer 2 is slower). 
        expIdx = []
        for t in range(int(args['timeSteps']/skipLen)):
            expIdx.append(np.zeros([skipLen])+t)
        expIdx = np.concatenate(expIdx).astype(int)
        self.logitOutput = tf.gather(self.logitOutput_downsample, expIdx, axis=1)

        #--------------Loss function--------------
        #here we accounting for the output delay
        labels = self.batchTargets[:,0:-(args['outputDelay']),:]
        logits = self.logitOutput[:,args['outputDelay']:,:]
        bw = self.batchWeight[:,0:-(args['outputDelay'])]

        #character transition signal is the last column, which has a separate loss
        transOut = logits[:,:,-1]
        transLabel = labels[:,:,-1]

        logits = logits[:,:,0:-1]
        labels = labels[:,:,0:-1]

        #cross-entropy character probability loss
        ceLoss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits)
        self.totalErr = tf.reduce_mean(tf.reduce_sum(bw*ceLoss,axis=1)/self.args['timeSteps'])

        #character start signal loss
        sqErrLoss = tf.square(tf.sigmoid(transOut)-transLabel)
        self.totalErr += 5*tf.reduce_mean(tf.reduce_sum(sqErrLoss,axis=1)/self.args['timeSteps'])

        #L2 regularizer
        weightVars = [self.readout_W]
        for inpIdx in range(self.nInpLayers):
            weightVars.append(self.inputFactors_W_all[inpIdx])
                
        weightVars.extend(self.rnnWeightVars)
        weightVars.extend(self.rnnWeightVars2)

        self.l2cost = tf.zeros(1,dtype=tf.float32)
        if self.args['l2scale']>0:
            for x in range(len(weightVars)):
                self.l2cost = self.l2cost + tf.reduce_sum(tf.square(weightVars[x]))

        #total cost
        self.totalCost = self.totalErr + self.l2cost*self.args['l2scale']

        #--------------Gradient descent--------------
        #prepare gradients and optimizer
        tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        
        #option to only allow the input layers to train
        if not bool(self.args['trainableBackEnd']):
            tvars.remove(self.rnnWeightVars[0])
            tvars.remove(self.rnnWeightVars2[0])
        
        #clip gradients to a maximum value of 10
        grads = tf.gradients(self.totalCost, tvars)
        grads, self.grad_global_norm = tf.clip_by_global_norm(grads, 10)

        #optimization routine & learning rate
        learnRate = tf.get_variable('learnRate',dtype=tf.float32,initializer=1.0,trainable=False)
        opt = tf.train.AdamOptimizer(learnRate, beta1=0.9, beta2=0.999,
                                     epsilon=1e-01)
        
        self.new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
        self.lr_update = tf.assign(learnRate, self.new_lr)
        
        #check if gradients are finite; if not, don't apply
        allIsFinite = []
        for g in grads:
            if g != None:
                allIsFinite.append(tf.reduce_all(tf.is_finite(g)))
        gradIsFinite = tf.reduce_all(tf.stack(allIsFinite))
        self.train_op = tf.cond(gradIsFinite, lambda: opt.apply_gradients(
            zip(grads, tvars), global_step=tf.train.get_or_create_global_step()), lambda: tf.no_op())

        #Initialize all variables in the model, potentially loading them if self.loadingInitParams==True
        self._loadAndInitializeVariables()
 
    def train(self):
        """
        The main training loop, which we have implemented manually here. Each loop makes a single call to sess.run to execute
        one minibatch. ALong the way, we periodically save the model and performance statistics.
        """
        saver = tf.train.Saver(max_to_keep=self.args['nCheckToKeep'])

        #Prepare to save performance data from each batch.
        batchTrainStats = np.zeros([self.args['nBatchesToTrain'],6])
        batchValStats = np.zeros([int(np.ceil(self.args['nBatchesToTrain']/self.args['batchesPerVal'])),4])
        i = self.startingBatchNum
        
        #load up previous statistics if we are resuming.
        if self.resumeTraining:
            resumedStats = scipy.io.loadmat(self.args['outputDir'] + '/intermediateOutput')
            batchTrainStats = resumedStats['batchTrainStats']
            batchValStats = resumedStats['batchValStats']
        
        #Save initial model parameters.
        saver.save(self.sess, self.args['outputDir'] + '/model.ckpt', global_step=0, write_meta_graph=False)
        
        #This ensures we aren't accidentally changing the graph as we go (which degrades performance).
        self.sess.graph.finalize()
            
        while i < self.args['nBatchesToTrain']:       
            #time how long this batch takes
            dtStart = datetime.now()

            #learn rate
            lr = self.args['learnRateStart']*(1 - i/float(self.args['nBatchesToTrain'])) + self.args['learnRateEnd']*(i/float(self.args['nBatchesToTrain']))

            #run train batch, selecting a day at random
            dayNum = np.argwhere(np.random.multinomial(1,self.dayProbability))[0][0]
            datasetNum = 2*dayNum #2*dayNum selects the train partition (as opposed to 2*dayNum + 1)
            
            runResultsTrain = self._runBatch(datasetNum=datasetNum, dayNum=dayNum, lr=lr, computeGradient=True, doGradientUpdate=True)
            
            #compute the frame-by-frame accuracy for this batch
            trainAcc = computeFrameAccuracy(runResultsTrain['logitOutput'], 
                                            runResultsTrain['targets'],
                                            runResultsTrain['batchWeight'], 
                                            self.args['outputDelay'])

            #record useful statistics about this minibatch
            totalSeconds = (datetime.now()-dtStart).total_seconds()
            batchTrainStats[i,:] = [i, runResultsTrain['err'], runResultsTrain['gradNorm'], trainAcc, totalSeconds, dayNum]

            #every once in a while, run a validation batch (i.e., run the RNN on the test partition to see how we're doing)
            if i%self.args['batchesPerVal']==0:
                valSetIdx = int(i/self.args['batchesPerVal'])
                batchValStats[valSetIdx,0:4], outputSnapshot = self._validationDiagnostics(i, self.args['batchesPerVal'], lr, 
                                                                                          totalSeconds, runResultsTrain, trainAcc)
                
                #save a snapshot of key RNN outputs/variables so an outside program can plot them if desired
                scipy.io.savemat(self.args['outputDir']+'/outputSnapshot', outputSnapshot)
            
            #save performance statistics and model parameters every so often
            if i>=(self.startingBatchNum+self.args['batchesPerSave']-1) and i%self.args['batchesPerSave']==0:
                scipy.io.savemat(self.args['outputDir'] + '/intermediateOutput', {'batchTrainStats': batchTrainStats,
                                                           'batchValStats': batchValStats})

            if i%self.args['batchesPerModelSave']==0:
                print('SAVING MODEL')
                saver.save(self.sess, self.args['outputDir'] + '/model.ckpt', global_step=i, write_meta_graph=False)
                    
            i += 1
                    
        #save final training statistics over all batches & final model
        scipy.io.savemat(self.args['outputDir'] + '/finalOutput', {'batchTrainStats': batchTrainStats,
                                                                   'batchValStats': batchValStats})

        print('SAVING FINAL MODEL')
        saver.save(self.sess, self.args['outputDir'] + '/model.ckpt', global_step=i, write_meta_graph=False)
    
    def inference(self):
        """
        Runs the RNN on the entire dataset once and returns the result - used at inference time for performance evaluation.
        """
                    
        #Compute how many total batches we'll need to run through before we go through everything once
        self.nBatchesForInference = np.ceil(self.nTrialsInFirstDataset / self.args['batchSize']).astype(int)
            
        #run through the entire dataset once
        allOutputs = []
        allUnits = []
        allInputFeatures = []

        print('Starting inference.')
        
        for x in range(self.nBatchesForInference):
            returnDict = self._runBatch(datasetNum=0, dayNum=0, lr=0, computeGradient=False, doGradientUpdate=False)        

            allOutputs.append(returnDict['logitOutput'])
            allInputFeatures.append(returnDict['inputFeatures'])
            allUnits.append(returnDict['output'])
        
        print('Done with inference.')
        
        #concatenate all batches and return
        allOutputs = np.concatenate(allOutputs,axis=0)
        allUnits = np.concatenate(allUnits,axis=0)
        allInputFeatures = np.concatenate(allInputFeatures,axis=0)
        
        #trim to original size in case the total number of sentences is not a multiple of the batch size
        allOutputs = allOutputs[0:self.nTrialsInFirstDataset,:,:]
        allUnits = allUnits[0:self.nTrialsInFirstDataset,:,:]
        allInputFeatures = allInputFeatures[0:self.nTrialsInFirstDataset,:,:]
                        
        retDict = {}
        retDict['outputs'] = allOutputs
        retDict['units'] = allUnits
        retDict['inputFeatures'] = allInputFeatures
        
        saveDict = {}
        saveDict['outputs'] = allOutputs
        
        if self.args['inferenceOutputFileName']!='None':
            scipy.io.savemat(self.args['inferenceOutputFileName'], saveDict)
        
        return retDict
    
    def _validationDiagnostics(self, i, nBatchesPerVal, lr, totalSeconds, runResultsTrain, trainAcc):
        """
        Runs a single minibatch on the validation data and returns performance statistics and a snapshot of key variables for
        diagnostic purposes. The snapshot file can be loaded and plotted by an outside program for real-time feedback of how
        the training process is going.
        """
        #Randomly select a day that has validation data; if there is no validation data, then just use the last days' training data
        if self.daysWithValData==[]:
            dayNum = self.nDays-1
            datasetNum = dayNum*2
        else:
            randIdx = np.random.randint(len(self.daysWithValData))
            dayNum = self.daysWithValData[randIdx]
            datasetNum = 1+dayNum*2 #odd numbers are the validation partitions
                
        runResults = self._runBatch(datasetNum=datasetNum, dayNum=dayNum, lr=lr, computeGradient=True, doGradientUpdate=False)
        
        valAcc = computeFrameAccuracy(runResults['logitOutput'], 
                                runResults['targets'],
                                runResults['batchWeight'], 
                                self.args['outputDelay'])

        print('Val Batch: ' + str(i) + '/' + str(self.args['nBatchesToTrain']) + ', valErr: ' + str(runResults['err']) + ', trainErr: ' + str(runResultsTrain['err']) + ', Val Acc.: ' + str(valAcc) + ', Train Acc.: ' + str(trainAcc) + ', grad: ' + str(runResults['gradNorm']) + ', learnRate: ' + str(lr) + ', time: ' + str(totalSeconds))
        
        outputSnapshot = {}
        outputSnapshot['inputs'] = runResults['inputFeatures'][0,:,:]
        outputSnapshot['rnnUnits'] = runResults['output'][0,:,:]
        outputSnapshot['charProbOutput'] = runResults['logitOutput'][0,:,0:-1]
        outputSnapshot['charStartOutput'] = scipy.special.expit(runResults['logitOutput'][0,self.args['outputDelay']:,-1])
        outputSnapshot['charProbTarget'] = runResults['targets'][0,:,0:-1]
        outputSnapshot['charStartTarget'] = runResults['targets'][0,:,-1]
        outputSnapshot['errorWeight'] = runResults['batchWeight'][0,:]
        
        return [i, runResults['err'], runResults['gradNorm'], valAcc], outputSnapshot
                
    def _runBatch(self, datasetNum, dayNum, lr, computeGradient, doGradientUpdate): 
        """
        Makes a single call to sess.run to execute one minibatch. Note that datasetNum and dayNum must be specified so we know
        which dataset to pull from (datasetNum) and which input layer to use (dayNum). 
        """
        fd = {self.new_lr: lr, self.datasetNumPH: int(datasetNum), self.dayNumPH: int(dayNum)}
        runOps = [self.totalErr, self.batchInputs, self.rnnOutput, self.batchTargets, self.logitOutput, self.batchWeight]  
        opMap = {}
        
        if computeGradient:
            runOps.extend([self.grad_global_norm])
            opMap['gradNorm'] = len(runOps)-1
            
        if doGradientUpdate:
            runOps.extend([self.lr_update, self.train_op])

        runResult = self.sess.run(runOps, feed_dict=fd)

        returnDict = {}
        returnDict['err'] = runResult[0]
        returnDict['inputFeatures'] = runResult[1]
        returnDict['output'] = runResult[2]
        returnDict['targets'] = runResult[3]
        returnDict['logitOutput'] = runResult[4]
        returnDict['batchWeight'] = runResult[5]
               
        if computeGradient:
            returnDict['gradNorm'] = runResult[opMap['gradNorm']]
        else:
            returnDict['gradNorm'] = 0

        return returnDict
    
    def _loadAndInitializeVariables(self):
        """
        Initializes all tensorflow variables on the graph, optionally loading their values from a specified file.
        """
        if self.loadingInitParams:
            #find the variables in the checkpoint
            ckpt = tf.train.get_checkpoint_state(self.args['loadDir'])
            loadCheckPointIdx = self.args['loadCheckpointIdx']
            checkpoint_name = os.path.basename(os.path.normpath(ckpt.all_model_checkpoint_paths[loadCheckPointIdx]))
            checkpoint_path = self.args['loadDir'] + '/' + checkpoint_name

            #print variables in the checkpoint
            print('Loading from checkpoint: ' + checkpoint_path)
            from tensorflow.contrib.framework.python.framework import checkpoint_utils

            var_list_ckpt = checkpoint_utils.list_variables(checkpoint_path)
            var_names_ckpt = []
            for v in var_list_ckpt: 
                var_names_ckpt.append(v[0])
                #print(v)

            #put together what variables we are going to load from what sources,
            #with special attention to how the inputFactors are determined
            lv = [self.readout_W, self.readout_b, self.rnnWeightVars[0], self.rnnWeightVars2[0], self.rnnStartState]
            varDict = {}
            for x in range(len(lv)):
                varDict[lv[x].name[:-2]] = lv[x]

            if self.args['mode']=='infer':
                varDict['inputFactors_W_'+str(self.args['inferenceInputLayer'])] = self.inputFactors_W_all[0]
                varDict['inputFactors_b_'+str(self.args['inferenceInputLayer'])] = self.inputFactors_b_all[0] 
                saver = tf.train.Saver(varDict)
                lastLayerSavers = []
            else:
                lastAvailableInpLayer = -1
                for inpLayerIdx in range(self.nInpLayers):
                    if 'inputFactors_W_'+str(inpLayerIdx) in var_names_ckpt:
                        lastAvailableInpLayer = inpLayerIdx
                        varDict['inputFactors_W_'+str(inpLayerIdx)] = self.inputFactors_W_all[inpLayerIdx]
                        varDict['inputFactors_b_'+str(inpLayerIdx)] = self.inputFactors_b_all[inpLayerIdx]    

                saver = tf.train.Saver(varDict)

                lastLayerSavers = []
                for inpLayerIdx in range(lastAvailableInpLayer+1, self.nInpLayers):
                    newDict = {}
                    newDict['inputFactors_W_'+str(lastAvailableInpLayer)] = self.inputFactors_W_all[inpLayerIdx]
                    newDict['inputFactors_b_'+str(lastAvailableInpLayer)] = self.inputFactors_b_all[inpLayerIdx] 
                    lastLayerSavers.append(tf.train.Saver(newDict))

        self.sess.run(tf.global_variables_initializer())
        self.startingBatchNum = 0
        if self.loadingInitParams:
            saver.restore(self.sess, checkpoint_path)
            for s in lastLayerSavers:
                s.restore(self.sess, checkpoint_path)
                
            if self.resumeTraining:
                self.startingBatchNum = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
                self.startingBatchNum += 1
                                 
    def _loadAllDatasets(self):
        """
        Loads the labels & data for each day specified in the training args, and returns the relevant variables as data cubes.
        Also collects the file names of all .tfrecord files needed for including the synthetic data.
        """
        neuralCube_all = []
        targets_all = []
        errWeights_all = []
        numBinsPerTrial_all = []
        cvIdx_all = []
        recordFileSet_all = []
    
        for dayIdx in range(self.nDays):
            neuralData, targets, errWeights, binsPerTrial, cvIdx = prepareDataCubesForRNN(self.args['sentencesFile_'+str(dayIdx)],
                                                                                          self.args['singleLettersFile_'+str(dayIdx)],
                                                                                          self.args['labelsFile_'+str(dayIdx)],
                                                                                          self.args['cvPartitionFile_'+str(dayIdx)],
                                                                                          self.args['sessionName_'+str(dayIdx)],
                                                                                          self.args['rnnBinSize'],
                                                                                          self.args['timeSteps'],
                                                                                          self.isTraining)

            neuralCube_all.append(neuralData)
            targets_all.append(targets)
            errWeights_all.append(errWeights)
            numBinsPerTrial_all.append(binsPerTrial)
            cvIdx_all.append(cvIdx)

            synthDir = self.args['syntheticDatasetDir_'+str(dayIdx)]
            if os.path.isdir(synthDir):
                recordFileSet = [os.path.join(synthDir, file) for file in os.listdir(synthDir)]
            else:
                recordFileSet = []
            
            if self.args['synthBatchSize']>0 and len(recordFileSet)==0:
                sys.exit('Error! No synthetic files found in directory ' + self.args['syntheticDatasetDir_'+str(dayIdx)] + ', exiting.')
                         
            random.shuffle(recordFileSet)
            recordFileSet_all.append(recordFileSet)
                                 
        return neuralCube_all, targets_all, errWeights_all, numBinsPerTrial_all, cvIdx_all, recordFileSet_all                          
    
    def _makeTrainingDatasetFromRealData(self, inputs, targets, errWeight, numBinsPerTrial, batchSize, addNoise=True):
        """
        This function creates a tensorflow 'dataset' from the real data given as input. Implements random
        extraction of data snippets from the full sentences and the optional addition of training noise of various kinds.
        
        Args:
            inputs (tensor : B x T x N): A 3d tensor of RNN inputs with batch size B, time steps T, and number of input features N
            targets (tensor : B x T x C): A 3d tensor of RNN targets with batch size B, time steps T, and number of targets C
            errWeight (tensor : B x T): A 2d tensor of error weights for each time step of data
            numBinsPerTrial (tensor : B): A 1d tensor describing the true length of each sentence in the batch (data is zero-padded)
            batchSize (int): Size of the mini-batch to construct

        Returns:
            iterator (tensorflow iterator): A dataset iterator that can be used to pull new minibatches
        """
        newDataset = tf.data.Dataset.from_tensor_slices((inputs.astype(np.float32), 
                                                         targets.astype(np.float32), 
                                                         errWeight.astype(np.float32), 
                                                         numBinsPerTrial.astype(np.int32)))

        newDataset = newDataset.apply(tf.contrib.data.shuffle_and_repeat(batchSize*4))

        mapFun = lambda inputs, targets, errWeight, numBinsPerTrial: extractSentenceSnippet(inputs, 
                                                                                             targets, 
                                                                                             errWeight, 
                                                                                             numBinsPerTrial, 
                                                                                             self.args['timeSteps'], 
                                                                                             self.args['directionality'])
        newDataset = newDataset.map(mapFun)

        if addNoise and (self.args['constantOffsetSD']>0 or self.args['randomWalkSD']>0):
            mapFun = lambda inputs, targets, errWeight, numBinsPerTrial: addMeanNoise(inputs, 
                                                                                      targets, 
                                                                                      errWeight, 
                                                                                      numBinsPerTrial, 
                                                                                      self.args['constantOffsetSD'], 
                                                                                      self.args['randomWalkSD'],
                                                                                      self.args['timeSteps'])
            newDataset = newDataset.map(mapFun)    

        if addNoise and self.args['whiteNoiseSD']>0:
            mapFun = lambda inputs, targets, errWeight, numBinsPerTrial: addWhiteNoise(inputs, 
                                                                                       targets, 
                                                                                       errWeight, 
                                                                                       numBinsPerTrial, 
                                                                                       self.args['whiteNoiseSD'],
                                                                                       self.args['timeSteps'])
            newDataset = newDataset.map(mapFun)    

        newDataset = newDataset.batch(batchSize)
        newDataset = newDataset.prefetch(1)

        iterator = newDataset.make_initializable_iterator()
        self.sess.run(iterator.initializer)

        return iterator
                        
def extractSentenceSnippet(inputs, targets, errWeight, numBinsPerTrial, nSteps, directionality):
    """
    Extracts a random snippet of data from the full sentence to use for the mini-batch.
    """          
    randomStart = tf.random.uniform(
                                [],
                                minval=0,
                                maxval=tf.maximum(numBinsPerTrial[0]+(nSteps-100)-400, 1),
                                dtype=tf.dtypes.int32)
    
    inputsSnippet = inputs[randomStart:(randomStart+nSteps),:]
    targetsSnippet = targets[randomStart:(randomStart+nSteps),:]
    
    charStarts = tf.where_v2(targetsSnippet[1:,-1] - targetsSnippet[0:-1,-1]>=0.1)        
    
    def noLetters():
        ews =  tf.zeros(shape=[nSteps])
        return ews

    def atLeastOneLetter():
        firstChar = tf.cast(charStarts[0,0], dtype=tf.int32)
        lastChar = tf.cast(charStarts[-1,0], dtype=tf.int32)
        
        if directionality=='unidirectional':
            #if uni-directional, only need to blank out the first part because it's causal with a delay
            ews =  tf.concat([tf.zeros(shape=[firstChar]), 
                              errWeight[(randomStart+firstChar):(randomStart+nSteps)]], axis=0)
        else:
            #if bi-directional (acausal), we need to blank out the last incomplete character as well so that only fully complete
            #characters are included
            ews =  tf.concat([tf.zeros(shape=[firstChar]), 
                              errWeight[(randomStart+firstChar):(randomStart+lastChar)],
                              tf.zeros(shape=[nSteps-lastChar])], axis=0)
            
        return ews
        
    errWeightSnippet = tf.cond(tf.equal(tf.shape(charStarts)[0], 0), 
                               noLetters, 
                               atLeastOneLetter)

    return inputsSnippet, targetsSnippet, errWeightSnippet, numBinsPerTrial
        
def addMeanNoise(inputs, targets, errWeight, numBinsPerTrial, constantOffsetSD, randomWalkSD, nSteps):
    """
    Applies mean drift noise to each time step of the data in the form of constant offsets (sd=sdConstant)
    and random walk noise (sd=sdRandomWalk)
    """  
    meanDriftNoise = tf.random_normal([1, int(inputs.shape[1])], mean=0, stddev=constantOffsetSD)
    meanDriftNoise += tf.cumsum(tf.random_normal([nSteps, int(inputs.shape[1])], mean=0, stddev=randomWalkSD), axis=1)
    
    return inputs+meanDriftNoise, targets, errWeight, numBinsPerTrial

def addWhiteNoise(inputs, targets, errWeight, numBinsPerTrial, whiteNoiseSD, nSteps):
    """
    Applies white noise to each time step of the data (sd=whiteNoiseSD)
    """
    whiteNoise = tf.random_normal([nSteps, int(inputs.shape[1])], mean=0, stddev=whiteNoiseSD)
                               
    return inputs+whiteNoise, targets, errWeight, numBinsPerTrial

def parseDataset(singleExample, nSteps, nInputs, nClasses, whiteNoiseSD=0.0, constantOffsetSD=0.0, randomWalkSD=0.0):
    """
    Parsing function for the .tfrecord file synthetic data. Returns a synthetic snippet with added noise for training.
    """
    features = {"inputs": tf.FixedLenFeature((nSteps, nInputs), tf.float32),
                "labels": tf.FixedLenFeature((nSteps, nClasses), tf.float32),
                "errWeights": tf.FixedLenFeature((nSteps), tf.float32),}
    parsedFeatures = tf.parse_single_example(singleExample, features)
        
    noise = tf.random_normal([nSteps, nInputs], mean=0.0, stddev=whiteNoiseSD)
    
    if constantOffsetSD>0 or randomWalkSD>0:
        trainNoise_mn = tf.random_normal([1, nInputs], mean=0, stddev=constantOffsetSD)
        trainNoise_mn += tf.cumsum(tf.random_normal([nSteps, nInputs], mean=0, stddev=randomWalkSD), axis=1)
        noise += trainNoise_mn
        
    return parsedFeatures["inputs"]+noise, parsedFeatures["labels"], parsedFeatures["errWeights"]

def cudnnGraphSingleLayer(nUnits, initRNNState, batchInputs, nSteps, nBatch, nInputs, direction):
    """
    Construct a single GRU layer using tensorflow cudnn calls for speed and define the shape before runtime. 
    Also return the weights so we can do L2 regularization.
    """
    nLayers = 1
    rnn_cudnn = tf.contrib.cudnn_rnn.CudnnGRU(nLayers, 
                                              nUnits, 
                                              dtype=tf.float32, 
                                              bias_initializer=tf.constant_initializer(0.0), 
                                              direction=direction)
    
    inputSize = [nSteps, nBatch, nInputs]
    rnn_cudnn.build(inputSize)
    
    #taking care to transpose the inputs and outputs for compatability with the rest of the code which is batch-first
    cudnnInput = tf.transpose(batchInputs,[1,0,2])
    y_cudnn, state_cudnn = rnn_cudnn(cudnnInput, initial_state=(initRNNState,))
    y_cudnn = tf.transpose(y_cudnn,[1,0,2])
    
    return y_cudnn, [rnn_cudnn.weights[0]]
                    
def gaussSmooth(inputs, kernelSD):
    """
    Applies a 1D gaussian smoothing operation with tensorflow to smooth the data along the time axis.
    
    Args:
        inputs (tensor : B x T x N): A 3d tensor with batch size B, time steps T, and number of features N
        kernelSD (float): standard deviation of the Gaussian smoothing kernel
        
    Returns:
        smoothedData (tensor : B x T x N): A smoothed 3d tensor with batch size B, time steps T, and number of features N
    """
                                 
    #get gaussian smoothing kernel
    inp = np.zeros([100])
    inp[50] = 1
    gaussKernel = gaussian_filter1d(inp, kernelSD)

    validIdx = np.argwhere(gaussKernel>0.01)
    gaussKernel = gaussKernel[validIdx]
    gaussKernel = np.squeeze(gaussKernel/np.sum(gaussKernel))
    
    #apply the convolution separately for each feature
    convOut = []
    for x in range(inputs.get_shape()[2]):
        convOut.append(tf.nn.conv1d(inputs[:,:,x,tf.newaxis], gaussKernel[:,np.newaxis,np.newaxis].astype(np.float32), 1, 'SAME'))
        
    #gather the separate convolutions together into a 3d tensor again
    smoothedData = tf.concat(convOut, axis=2)
    
    return smoothedData

def computeFrameAccuracy(rnnOutput, targets, errWeight, outputDelay):
    """
    Computes a frame-by-frame accuracy percentage given the rnnOutput and the targets, while ignoring
    frames that are masked-out by errWeight and accounting for the RNN's outputDelay. 
    """
    #Select all columns but the last one (which is the character start signal) and align rnnOutput to targets
    #while taking into account the output delay. 
    bestClass = np.argmax(rnnOutput[:,outputDelay:,0:-1], axis=2)
    indicatedClass = np.argmax(targets[:,0:-outputDelay,0:-1], axis=2)
    bw = errWeight[:,0:-outputDelay]

    #Mean accuracy is computed by summing number of accurate frames and dividing by total number of valid frames (where bw == 1)
    acc = np.sum(bw*np.equal(np.squeeze(bestClass), np.squeeze(indicatedClass)))/np.sum(bw)
    
    return acc

def getDefaultRNNArgs():
    """
    Makes a default 'args' dictionary with all RNN hyperparameters populated with default values.
    """
    args = {}

    #These arguments define each dataset that will be used for training.
    rootDir = '/home/fwillett/handwritingDatasetsForRelease/'
    dataDirs = ['t5.2019.05.08']
    cvPart = 'HeldOutBlocks'

    for x in range(len(dataDirs)):
        args['timeSeriesFile_'+str(x)] = rootDir+'Step2_HMMLabels/'+cvPart+'/'+dataDirs[x]+'_timeSeriesLabels.mat'
        args['syntheticDatasetDir_'+str(x)] = rootDir+'Step3_SyntheticSentences/'+cvPart+'/'+dataDirs[x]+'_syntheticSentences/'
        args['cvPartitionFile_'+str(x)] = rootDir+'trainTestPartitions_'+cvPart+'.mat'
        args['sessionName_'+str(x)] = dataDirs[x]

    #Specify which GPU to use (on multi-gpu machines, this prevents tensorflow from taking over all GPUs)
    args['gpuNumber'] = '0'
    
    #mode can either be 'train' or 'inference'
    args['mode'] = 'train'
    
    #where to save the RNN files
    args['outputDir'] = rootDir+'Step4_RNNTraining/'+cvPart
    
    #We can load the variables from a previous run, either to resume training (if loadDir==outputDir) 
    #or otherwise to complete an entirely new training run. 'loadCheckpointIdx' specifies which checkpoint to load (-1 = latest)
    args['loadDir'] = 'None'
    args['loadCheckpointIdx'] = -1
    
    #number of units in each GRU layer
    args['nUnits'] = 512
    
    #Specifies how many 10 ms time steps to combine a single bin for RNN processing                              
    args['rnnBinSize'] = 2
    
    #Applies Gaussian smoothing if equal to 1                             
    args['smoothInputs'] = 1
    
    #For the top GRU layer, how many bins to skip for each update (the top layer runs at a slower frequency)                             
    args['skipLen'] = 5
    
    #How many bins to delay the output. Some delay is needed in order to give the RNN enough time to see the entire character
    #before deciding on its identity. Default is 1 second (50 bins).
    args['outputDelay'] = 50 
    
    #Can be 'unidrectional' (causal) or 'bidirectional' (acausal)                              
    args['directionality'] = 'unidirectional'

    #standard deivation of the constant-offset firing rate drift noise                             
    args['constantOffsetSD'] = 0.6
    
    #standard deviation of the random walk firing rate drift noise                             
    args['randomWalkSD'] = 0.02 
   
    #standard deivation of the white noise added to the inputs during training                            
    args['whiteNoiseSD'] = 1.2
    
    #l2 regularization cost                             
    args['l2scale'] = 1e-5 
                                
    args['learnRateStart'] = 0.01
    args['learnRateEnd'] = 0.0
    
    #can optionally specify for only the input layers to train or only the back end                             
    args['trainableInput'] = 1
    args['trainableBackEnd'] = 1

    #this seed is set for numpy and tensorflow when the class is initialized                             
    args['seed'] = datetime.now().microsecond

    #number of checkpoints to keep saved during training                             
    args['nCheckToKeep'] = 1
    
    #how often to save performance statistics                              
    args['batchesPerSave'] = 200
                                 
    #how often to run a validation diagnostic batch                              
    args['batchesPerVal'] = 50
                                 
    #how often to save the model                             
    args['batchesPerModelSave'] = 5000
                                 
    #how many minibatches to use total                             
    args['nBatchesToTrain'] = 100000 

    #number of time steps to use in the minibatch (1200 = 24 seconds)                             
    args['timeSteps'] = 1200 
                                 
    #number of sentence snippets to include in the minibatch                             
    args['batchSize'] = 64 
                                 
    #how much of each minibatch is synthetic data                              
    args['synthBatchSize'] = 24 

    #can be used to scale up all input features, sometimes useful when transferring to new days without retraining 
    args['inputScale'] = 1.0
                                 
    #parameters to specify where to save the outputs and which layer to use during inference                             
    args['inferenceOutputFileName'] = 'None'
    args['inferenceInputLayer'] = 0

    #defines the mapping between each day and which input layer to use for that day                             
    args['dayToLayerMap'] = '[0]'
                                 
    #for each day, the probability that a minibatch will pull from that day. Can be used to weight some days more than others  
    args['dayProbability'] = '[1.0]'

    return args
    
#Here we provide support for running from the command line.
#The only command line argument is the name of an args file.
#Launching from the command line is more reliable than launching from within a jupyter notebook, which sometimes hangs.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--argsFile', metavar='argsFile', 
                        type=str, default='args.p')

    args = parser.parse_args()
    args = vars(args)   
    argDict = pickle.load( open( args['argsFile'], "rb" ) )

    #set the visible device to the gpu specified in 'args' (otherwise tensorflow will steal all the GPUs)
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    print('Setting CUDA_VISIBLE_DEVICES to ' + argDict['gpuNumber'])
    os.environ["CUDA_VISIBLE_DEVICES"]=argDict['gpuNumber']
    
    #instantiate the RNN model
    rnnModel = charSeqRNN(args=argDict)

    #train or infer
    if argDict['mode']=='train':
        rnnModel.train()
    elif argDict['mode']=='inference':
        rnnModel.inference()