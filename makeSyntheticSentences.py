import numpy as np
import scipy.io
import tensorflow as tf
from datetime import datetime

def generateCharacterSequences(args):
    """
    Generates synthetic data by taking character snippets from a library and arranging them into
    random sentences.
    
    Args:
        args (dict): An arguments dictionary with the following entries:
            charDef (dict): A definition of character names and lengths (see characterDefinitions.py)
            snippetFile (str): A file name pointing to a snippet library file (created in Step 2)
            nSentences (int): Number of sentences to generate
            nSteps (int): Number of time steps to generate per sentence
            binSize (int): Optionally bin the data if binSize>1
            saveFile (str): Name of the .tfrecord file we will save the synthetic data to
            wordListFile (str): A list of valid words to use when randomly generating sentences
            rareWordFile (str): Name of a file containing a list of indices pointing to the words in 'wordList' 
                                that contain rare letters ('x','z','q','j'). Can set to 'None' to turn off this feature.
            accountForPenState (bool): If true, attempt to respect pen transition movements by choosing character snippets that
                                       end with the pen in the correct place to begin the next character.
    Returns:
        synthNeuralBinned (matrix : N x T x E): A tensor of synthetic neural data 
                                          (N = # of sentences, T = # of time bins, E = # of electrodes)      
        synthTargetsBinned (matrix : N x T x C+1): A tensor of character probability targets and character start signal targets
                                                   (last column) (N = # of sentences, T = # of time bins, C = # of characters)             
    """
        
    #set unique seed
    np.random.seed(args['seed'])

    #load snippet library
    charSnippets = scipy.io.loadmat(args['snippetFile'])
    
    #add extra steps so we can take random snippets from the finished product, like we will for the real data
    nStepsToGenerate = args['nSteps'] + 2000
    
    #We use a 'rare' word list file to increase the frequency of words with rare letters ('z', 'x', 'j', 'q').
    #The rare word file contains the indices of the words in 'wordListFile' with rare letters.
    wordList = [line.rstrip('\n') for line in open(args['wordListFile'])]
    if args['rareWordFile']!='None':
        rareWordFile = scipy.io.loadmat(args['rareWordFile'])
        rareWordList = (np.squeeze(rareWordFile['rareIdx'])-1).tolist()
        rareLetterIncrease = True
    else:
        rareLetterIncrease = False
        
    #generate synthetic sentences
    synthNeural, synthCharProb, synthCharStart = makeSyntheticDataFromRawSnippets(charDef = args['charDef'], 
                                                                                  charSnippets = charSnippets, 
                                                                                  nSentences = args['nSentences'], 
                                                                                  nSteps = nStepsToGenerate, 
                                                                                  wordList = wordList, 
                                                                                  blankProb = 0.20,
                                                                                  accountForPenState = args['accountForPenState'],
                                                                                  rareLetterIncrease = rareLetterIncrease, 
                                                                                  rareWordList = rareWordList) 

    #combine character probabilities with character transition signal
    synthTargets = np.concatenate([synthCharProb, synthCharStart[:,:,np.newaxis]], axis=2)
    
    #cut off the first part of the data so the RNN starts off "hot" randomly in the middle of text
    synthNeural_cut = np.zeros([args['nSentences'], args['nSteps'], synthNeural.shape[2]])
    synthTargets_cut = np.zeros([args['nSentences'], args['nSteps'], synthTargets.shape[2]])

    for t in range(args['nSentences']):
        randStart = np.random.randint(nStepsToGenerate-args['nSteps'])
        synthNeural_cut[t,:,:] = synthNeural[t,randStart:(randStart+args['nSteps']),:]
        synthTargets_cut[t,:,:] = synthTargets[t,randStart:(randStart+args['nSteps']),:]

    synthNeural = synthNeural_cut
    synthTargets = synthTargets_cut

    #bin the data
    if args['binSize']==1:
        synthNeuralBinned = synthNeural
        synthTargetsBinned = synthTargets
    else:
        nBins = np.ceil(args['nSteps']/args['binSize']).astype(int)
        synthNeuralBinned = np.zeros([args['nSentences'], nBins, synthNeural.shape[2]])
        synthTargetsBinned = np.zeros([args['nSentences'], nBins, synthTargets.shape[2]])
        
        currIdx = np.arange(0,args['binSize']).astype(int)
        for x in range(nBins):
            synthNeuralBinned[:,x,:] = np.mean(synthNeural[:,currIdx,:],axis=1)
            synthTargetsBinned[:,x,:] = np.mean(synthTargets[:,currIdx,:],axis=1)
            currIdx += args['binSize']
            
    #create an error mask that doesn't penalize the RNN for errors that occur before the first character starts
    errWeights = np.ones([synthTargetsBinned.shape[0], synthTargetsBinned.shape[1]])
    for t in range(errWeights.shape[0]):
        charStarts = np.argwhere(synthTargetsBinned[t,1:,-1]-synthTargetsBinned[t,0:-1,-1]>=0.1).astype(np.int32)
        if len(charStarts)==0:
            errWeights[t,:] = 0
        else:
            errWeights[t,0:charStarts[0,0]] = 0
          
    #store the sentences in a .tfrecord file so tensorflow can read them quickly during RNN training
    def _floats_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))
    
    writer = tf.python_io.TFRecordWriter(args['saveFile'])
    for trialIdx in range(args['nSentences']):
        feature = {'inputs': _floats_feature(np.ravel(synthNeuralBinned[trialIdx,:,:]).tolist()),
            'labels': _floats_feature(np.ravel(synthTargetsBinned[trialIdx,:,:]).tolist()),
            'errWeights': _floats_feature(np.ravel(errWeights[trialIdx,:]).tolist())}

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())

    writer.close()
    
    return synthNeuralBinned, synthTargetsBinned, errWeights

def makeSyntheticDataFromRawSnippets(charDef, charSnippets, nSentences, nSteps, wordList, blankProb=0.05, accountForPenState=True, rareLetterIncrease=False, rareWordList=[]):
    """
    Generates synthetic data by taking character snippets from the library 'charSnippets' and arrnaging them into
    random sentences.
    
    Args:
        charDef (dict): A definition of character names and lengths (see characterDefinitions.py)
        charSnippets (dict): A library of neural data snippets that correspond to single characters. These can be arranged
                             into random synthetic sentences. 
        nSentences (int): Number of sentences to generate
        nSteps (int): Number of time steps to generate per sentence
        wordList (list): A list of valid words to use when randomly generating sentences
        blankProb (float): Probability of generating a 'blank' period that simulates the user taking a brief pause
        accountForPenState (bool): If true, attempt to respect pen transition movements by choosing character snippets that
                                   end with the pen in the correct place to begin the next character.
        rareLetterIncrease (bool): If true, increases the frequency of words with rare letters by choosing from 'rareWordList' 
                                 more often.
        rareWordList (list): A list of indices pointing to the words in 'wordList' that contain rare letters ('x','z','q','j')
                                   
    Returns:
        synthNeural (matrix : N x T x E): A tensor of synthetic neural data 
                                          (N = # of sentences, T = # of time steps, E = # of electrodes)      
        synthCharProb (matrix : N x T x C): A tensor of character probability targets
                                           (N = # of sentences, T = # of time steps, C = # of characters)             
        synthCharStart (matrix : N x T): A tensor of character start signal targets
                                         (N = # of sentences, T = # of time steps)        
    """

    nNeurons = charSnippets['a'][0,0].shape[1]
    nClasses = len(charDef['charList'])

    synthNeural = np.zeros([nSentences, nSteps, nNeurons])
    synthCharProb = np.zeros([nSentences, nSteps, nClasses])
    synthCharStart = np.zeros([nSentences, nSteps])
                                                                
    for t in range(nSentences):
        currIdx = 0
        currentWord = []
        currentLetterIdx = 0
        
        #generate this sentence one character at a time
        while currIdx<nSteps:
            #pick a new word if needed
            if currentLetterIdx>=len(currentWord):
                currentLetterIdx = 0
                currentWord = pickWordForSentence(wordList, rareLetterIncrease=rareLetterIncrease, rareWordList=rareWordList)

            #pick the character snippet to use for the current character
            classIdx = charDef['strToCharIdx'][currentWord[currentLetterIdx]]            
            if (currentLetterIdx<(len(currentWord)-1)) and accountForPenState:
                nextClassIdx = charDef['strToCharIdx'][currentWord[currentLetterIdx+1]]
                nextPenStartLoc = charDef['penStart'][nextClassIdx]
                
                #if possible, choose a letter snippet that ends with the pen set up for the next one
                penEndStates = charSnippets[charDef['charList'][classIdx]+'_penEndState']
                
                #-2 is a special code that means the snippet can always be used; useful when there isn't enough data
                validIdx = np.argwhere(np.logical_or(penEndStates[0,:]==nextPenStartLoc, penEndStates[0,:]<-1.5))
                
                if validIdx.shape[0]==0:
                    #choose a random one
                    choiceIdx = np.random.randint(len(charSnippets[charDef['charList'][classIdx]][0]))
                else:
                    #choose randomly from the valid ones
                    choiceIdx = np.random.randint(len(validIdx))
                    choiceIdx = validIdx[choiceIdx][0]
            else:
                #last letter or not accounting for pen state
                choiceIdx = np.random.randint(len(charSnippets[charDef['charList'][classIdx]][0]))
                    
            currentSnippet = charSnippets[charDef['charList'][classIdx]][0,choiceIdx].copy()
            useIdx = np.logical_not(np.isnan(currentSnippet[:,0]))
            currentSnippet = currentSnippet[useIdx,:]
            
            #linear time-warping & re-scaling to add more variability
            charLen = currentSnippet.shape[0]  
            nStepsForChar = np.round(charLen*0.7 + np.random.randint(charLen*0.6))

            tau = np.linspace(0, currentSnippet.shape[0]-1, int(nStepsForChar))
            tau = np.round(tau).astype(int)
            currentSnippet = currentSnippet[tau,:]
            
            randScale = 0.7 + 0.6*np.random.rand()
            currentSnippet *= randScale

            #randomly add in 'blank' pauses with some probability
            if np.random.rand(1)<blankProb:
                choiceIdx = np.random.randint(charSnippets['blank'].shape[1])
                blankData = charSnippets['blank'][0,choiceIdx]
                blankLen = blankData.shape[0]
                currentSnippet = np.concatenate([currentSnippet, blankData], axis=0)

            #generate probability targets for this character
            labels = np.zeros([currentSnippet.shape[0], nClasses])
            labels[:,classIdx] = 1

            #fill in the data tensors for this character
            nNewSteps = currentSnippet.shape[0]
            if nNewSteps+currIdx >= nSteps:
                stepLimit = nSteps - currIdx
                currentSnippet = currentSnippet[0:stepLimit,:]
                labels = labels[0:stepLimit,:]

            synthNeural[t,currIdx:(currIdx+currentSnippet.shape[0]),:] = currentSnippet
            synthCharProb[t,currIdx:(currIdx+currentSnippet.shape[0]),:] = labels
            synthCharStart[t,currIdx:(currIdx+20)] = 1
            
            #advance pointer to the next character
            currIdx += nNewSteps
            currentLetterIdx += 1
            
    return synthNeural, synthCharProb, synthCharStart

def pickWordForSentence(wordList, rareLetterIncrease=False, rareWordList=[]):
    """
    Implements a simple heuristic for randomly choosing which word to place next in the sentence.
    Each word is chosen independently of the previous words; the motivation was to prevent the RNN from learning
    a language model that extends beyond single words. 
    
    Args:
        wordList (list): A list of possible words.
        rareLetterIncrease (bool): If true, increases the frequency of words with rare letters by choosing from 'rareWordList' 
                                 more often.
        rareWordList (list): A list of indices pointing to the words in 'wordList' that contain rare letters ('x','z','q','j')
        
    Returns:
        nextWord (str): A string containing the randomly chosen word. 
    """
    
    #choose new word
    if np.random.rand()<0.2:
        #choose high frequency word
        wordIdx = np.random.randint(20)
    elif rareLetterIncrease and np.random.rand()<0.2:
        #choose a word with a rare letter in it ('x','z','q','j')
        rareIdx = np.random.randint(len(rareWordList))
        wordIdx = rareWordList[rareIdx]
    else:
        #choose any word
        wordIdx = np.random.randint(len(wordList))

    nextWord = list(wordList[wordIdx])

    #with low probability, place an apostrophe before the last character in the word
    if np.random.rand()<0.03 and len(nextWord)>3:
        nextWord.insert(len(nextWord)-1,"'")

    #with low probability, place a comma, period or question mark at the end of the word
    putComma = False
    putPeriod = False
    putQuestion = False

    if np.random.rand()<0.07:
        putComma = True
    elif np.random.rand()<0.05:
        putPeriod = True
    elif np.random.rand()<0.05:
        putQuestion = True

    if putComma:
        nextWord.extend(',')
    if putPeriod:
        nextWord.extend('~')
    if putQuestion:
        nextWord.extend('?')

    #add a space to the end of the word
    if not putPeriod and not putQuestion:
        nextWord.extend('>')
        
    return nextWord

def extractCharacterSnippets(letterStarts, blankWindows, neuralCube, sentences, sentenceLens, trainPartitionIdx, charDef):
    """
    Constructs the time series 'targets' used to train the RNN. The RNN is trained using supervised learning to 
    produce the following two outputs: a character probability vector with a one-hot encoding of the current character, 
    and a binary'new character' signal which briefly goes high at the start of any new character. 

    This function also produces an 'ignoreError' mask which describes, for each time step, whether the cost function should ignore
    any errors at that time step. We use this feature to prevent the RNN from being pnealized for errors that occur at the very
    start of the trial, before T5 has written any character yet (if the HMM has labeled this as a 'blank' state).
    
    Args:
        letterStarts (matrix : N x 200): A matrix of character start times, each row corresponds to a single sentence.
        blankWindows (list): A nested list of time windows where periods of 'blank' pauses occur in the data. This is used to extract
                             'blank' snippets for simulating pauses. 
        neuralCube (matrix : N x T x E): A normalized, smooth neural activity cube (N = # of sentences, T = # of time steps, 
                                         E = # of electrodes)
        sentences (vector : N x 1): An array of sentences
        sentenceLens (vector : N x 1): An array of sentence lengths (the number of time steps per sentence)
        trainPartitionIdx (vector : C x 1): A vector containing the index of each sentence that belongs to the training set
        charDef (dict): A definition of character names and lengths (see characterDefinitions.py)
        
    Returns:
        snippetDict (dict): A dictionary containing character snippets and, for each snippet, an estimate of where the pen tip ended
    """
    
    #cut out snippets for synthetic data generation
    snippetDict = {}
    
    #initialize the snippet dictionary
    for thisChar in charDef['charList']:
        snippetDict[thisChar] = []
        snippetDict[thisChar+'_penEndState'] = []
        
    snippetDict['blank'] = []

    #For each sentence, cut out all of its characters
    for sentIdx in range(len(sentences)):

        #ignore characters that are not in the training data
        if not np.any(trainPartitionIdx==sentIdx):
            continue

        #number of characters in this sentence
        nChars = len(sentences[sentIdx][0])

        #Cut out each character one at a time from this sentence.
        for x in range(nChars):

            #Take a snippet of data from the start of this letter to the start of the next one.
            #If we are at the end of the sentence, take all data until the end.
            if x<(nChars-1):
                loopIdx = np.arange(letterStarts[sentIdx,x], letterStarts[sentIdx,x+1]).astype(np.int32)
            else:
                loopIdx = np.arange(letterStarts[sentIdx,x], sentenceLens[sentIdx]).astype(np.int32)

            newCharDat = neuralCube[sentIdx, loopIdx, :]

            #estimate where the pen tip is at the end of this snippet, which we are assuming contains the transition 
            #movement to the next character (i.e., we assume each snippet contains the transition movement to the next character).
            if x<(nChars-1): 
                nextChar = sentences[sentIdx][0][x+1]
                nextCharIdx = np.squeeze(np.argwhere(np.array(charDef['charListAbbr'])==nextChar))
                nextCharStartPenState = charDef['penStart'][nextCharIdx]
            else:
                nextCharStartPenState = -1

            thisChar = sentences[sentIdx][0][x]
            thisCharIdx = np.squeeze(np.argwhere(np.array(charDef['charListAbbr'])==thisChar))
            thisCharName = charDef['charList'][thisCharIdx]
            
            snippetDict[thisCharName].append(newCharDat)
            snippetDict[thisCharName+'_penEndState'].append(nextCharStartPenState)   

        #cut out the snippets that the HMM has labeled as 'blank'
        bw = blankWindows[0,sentIdx]
        for b in range(len(bw)):
            snippetDict['blank'].append(neuralCube[sentIdx, bw[0,b][bw[0,b]<neuralCube.shape[1]], :])

    return snippetDict

def addSingleLetterSnippets(snippetDict, slDat, twCubes, charDef):
    """
    Constructs the time series 'targets' used to train the RNN. The RNN is trained using supervised learning to 
    produce the following two outputs: a character probability vector with a one-hot encoding of the current character, 
    and a binary'new character' signal which briefly goes high at the start of any new character. 

    This function also produces an 'ignoreError' mask which describes, for each time step, whether the cost function should ignore
    any errors at that time step. We use this feature to prevent the RNN from being pnealized for errors that occur at the very
    start of the trial, before T5 has written any character yet (if the HMM has labeled this as a 'blank' state).
    
    Args:
        snippetDict (dict): A dictionary containing character snippets and, for each snippet, an estimate of where the pen tip ended
        slDat (dict): Single letter data dictionary
        twCubes (dict): A dictionary of matrices defining the spatiotemporal pattern of neural activity for each character,
                  indexed by that character's string representation.
        charDef (dict): A definition of character names and lengths (see characterDefinitions.py)
        
    Returns:
        snippetDict (dict): A dictionary containing character snippets and, for each snippet, an estimate of where the pen tip ended
    """
    
    #add each character to the snippet library
    for charIdx in range(len(charDef['charList'])):
        char = charDef['charList'][charIdx]
        neuralCube = slDat['neuralActivityCube_'+char].astype(np.float64)

        #get the trials that belong to this character
        trlIdx = []
        for t in range(slDat['characterCues'].shape[0]):
            if slDat['characterCues'][t,0]==char:
                trlIdx.append(t)

        #get the block that each trial belonged to
        blockIdx = slDat['blockNumsTimeSeries'][slDat['goPeriodOnsetTimeBin'][trlIdx]]
        blockIdx = np.squeeze(blockIdx)

        #subtract block-specific means from each trial 
        for b in range(slDat['blockList'].shape[0]):
            trialsFromThisBlock = np.squeeze(blockIdx==slDat['blockList'][b])
            neuralCube[trialsFromThisBlock,:,:] -= slDat['meansPerBlock'][np.newaxis,b,:]

        #divide by standard deviation to normalize the units
        neuralCube = neuralCube / slDat['stdAcrossAllData'][np.newaxis,:,:]

        #add each example
        for t in range(neuralCube.shape[0]):
            endStep = np.argwhere(twCubes[char+'_T'][:,t]>=60+charDef['charLen'][charIdx]).astype(np.int32)
            if len(endStep)==0:
                continue

            newExample = neuralCube[t,60:endStep[0,0],:]
            snippetDict[char].append(newExample)
            snippetDict[char+'_penEndState'].append(-2)

    return snippetDict
