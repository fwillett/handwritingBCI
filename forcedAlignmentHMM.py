import numpy as np
import scipy.ndimage.filters
import sklearn.decomposition 

def makeForcedAlignmentHMM(templates, sentence, hmmBinSize, blankProb):
    """
    This function initializes the HMM state transition matrix and emission probabilities for a particular sentence.
    Each state corresponds to a piece of a character, and the states march forward through the sentence.
    The emission probabilities are defined by the 'templates', which are spatiotemporal patterns of expected 
    neural activity (mean firing rates) for each character.
    
    Args:
        templates (dict): A dictionary of matrices defining the spatiotemporal pattern of neural activity for each character,
                          indexed by that character's string representation. 
        sentence (str): The sentence to be labeled
        hmmBinSize (int): The HMM bin size (# of 10 ms bins to group into one state)
        blankProb (float): probability of entering a blank state at the end of a character (to model pauses between characters)

    Returns:
        A_hmm (matrix : S x S): The state transition matrix, where entry (i,j) describes the probability of going from state i to j
        B_hmm (matrix : S x N): The emission distributions, where each column i defines the mean firing rates for state i
        diagVariance (vector : N x 1): The variance of each electrode's firing rate (the diagonal of the multivariate covariance)
        stateLabels (vector : S x 1): Each element describes which character that state belongs to
        stateLabelsSeq (vector : S x 1): Each element describes the position of that state's character in the sentence
    """

    #probability of persisting in the same state
    stayProb = 0.20 
    
    #probability of skipping two states ahead (instead of simply advancing to the next one)
    skipProb = 0.20 

    #First, compute the total number of states in the HMM (nStates) and the state at which each letter begins (letterStartIdx)
    nStates = 0
    letterStartIdx = np.zeros(len(sentence))
    for x in range(len(sentence)):
        letterStartIdx[x] = nStates

        #+1 for blank state at the end of each character
        nStates = nStates + np.floor(templates[sentence[x]].shape[0]/hmmBinSize) + 1 

    #+1 for blank state at the beginning of the sentence
    nStates = (nStates + 1).astype(np.int32) 

    #adjust letterStartIdx to account for the blank state at the beginning of the sentence
    letterStartIdx += 1
    letterStartIdx = letterStartIdx.astype(np.int32)

    #Next, define the state transitions and emission distributions.
    #The A matrix defines the state transitions (entry [i,j] is the probability of moving from i->j).
    #The B matrix defines the emission probabilities (by defining mean firing rates for each state). One row for each state.
    A_hmm = np.zeros([nStates, nStates])
    B_hmm = np.zeros([nStates, templates[sentence[0]].shape[1]])

    #For each state, store which character it belongs to (stateLabels) and that character's position in the sentence (stateLabelsSeq)
    stateLabels = np.zeros([nStates,1], dtype='str')
    stateLabelsSeq = np.zeros([nStates,1])
    nChars = len(templates)

    #loop through each character in the sentence and add that character's states to the HMM
    for x in range(len(sentence)):

        #nBins is the number of HMM states in this character
        nBins = np.floor(templates[sentence[x]].shape[0]/hmmBinSize).astype(np.int32)

        #idxInTemplate keeps track of where we are in the character template for each HMM state
        idxInTemplate = np.arange(0, hmmBinSize).astype(np.int32)

        currentState = letterStartIdx[x]

        #loop through each HMM state belonging to the current character
        for b in range(nBins):

            #define the mean firing rates for this state
            meanRates = np.mean(templates[sentence[x]][idxInTemplate,:], axis=0)
            B_hmm[currentState, :] = meanRates

            #define which character and position this state belongs to
            stateLabels[currentState] = sentence[x]
            stateLabelsSeq[currentState] = x

            #define transition probabilities for this state
            A_hmm[currentState, currentState] = stayProb

            if b<(nBins-1):
                #--prior to last state--
                if b==(nBins-2):
                    #second to last state, so no skipping two states ahead
                    A_hmm[currentState, currentState+1] = 1-stayProb
                else:
                    #otherwise, can advance one state OR skip two states ahead
                    A_hmm[currentState, currentState+1] = 1-stayProb-skipProb
                    A_hmm[currentState, currentState+2] = skipProb
            else:
                #--last state--
                #we can either transition to a blank state at the end of the character or go to the next character

                #special blank state
                A_hmm[currentState, currentState+1] = (1-stayProb)*blankProb
                A_hmm[currentState+1, currentState+1] = 0.5

                stateLabels[currentState+1] = '&' #blank symbol
                stateLabelsSeq[currentState+1] = -1

                #go to the next character in the sentence (or end the sentence)
                if x<(len(sentence)-1):
                    #transition to next letter
                    A_hmm[currentState, letterStartIdx[x+1]] = (1-stayProb)*(1-blankProb) #last letter state
                    A_hmm[currentState+1, letterStartIdx[x+1]] = 0.5 #blank
                else:
                    #end of the sentence
                    A_hmm[currentState, currentState+1] = (1-stayProb)
                    A_hmm[currentState+1, currentState+1] = 1.0 #stay in blank permanently (this is the end of the sentence)

            currentState += 1
            idxInTemplate += hmmBinSize

    #define the beginning blank state
    stateLabels[0] = '&'
    stateLabelsSeq[0] = -1

    A_hmm[0,0] = 0.5
    A_hmm[0,1] = 0.5

    #fill in emission probabilities for all blank states
    letterStates = stateLabelsSeq >= 0
    blankStates = stateLabelsSeq ==-1
    blankFiringRates = np.mean(B_hmm[letterStates[:,0],:],axis=0,keepdims=True)

    B_hmm[blankStates[:,0],:] = blankFiringRates

    #define the variance for each emission dimension (assuming multivariate normal distribution with diagonal covariance)
    diagVariance = np.ones([1 ,B_hmm.shape[1]])
    
    return A_hmm, B_hmm, diagVariance, stateLabels, stateLabelsSeq

def makeTimeWindowMask(stateLabelsSeq, nTimeSteps):
    """
    For labeling long sentences, it can help to enforce that characters occur within a certain 
    time window defined by their location in the sentence. This prevents pathological solutions 
    that place a large chunk of characters very close together. This mask is also used to enforce 
    sentence termination (by allowing only the final two states to occur at the last time step).
    
    Args:
        stateLabelsSeq (vector : S x 1) : Returned by makeForcedAlignmentHMM. Each element describes the location of 
                                          that state's character in the sentence.
        nTimeSteps (int): Number of time bins in the sentence that we are labeling.

    Returns:
        timeWindowMask (matrix : T x S): A time window mask that is multiplied against the HMM observation probabilities 
            during forced alignment. If it's zero for a given state and time step, then that state cannot appear at that time step.
    """

    nStates = stateLabelsSeq.shape[0]
    nChar = np.max(stateLabelsSeq)
    timeWindowMask = np.zeros([nTimeSteps, nStates])

    #Determine how big of a window we give each character, depending on sentence length
    if nChar==1:
        winLen = 1
    elif nChar<=10:
        winLen = 0.50
    else:
        winLen = 0.30

    #Now fill in the mask one state at a time
    for x in range(nStates):
        #first, get this state's character-level position in the sentence
        sl = stateLabelsSeq[x]

        #special logic for blank states (-1) or termination state (-2)
        if sl==-1:
            sl = stateLabelsSeq[x-1]
        elif sl==-2:
            sl = stateLabelsSeq[x-2]
        
        #compute where this character should occur in normalized time
        timeFraction = sl/nChar

        #compute the window of allowable time steps that this character can appear
        tIdx = np.arange(np.round(timeFraction*nTimeSteps - winLen*nTimeSteps),
                         np.round(timeFraction*nTimeSteps + winLen*nTimeSteps)).astype(np.int32)
        tIdx = tIdx[np.logical_and(tIdx>=0, tIdx<nTimeSteps)]

        timeWindowMask[tIdx, x] = 1

    #sentence termination constraint (disallows all states at the last time step EXCEPT the final character state or blank)
    timeWindowMask[-1, 0:-2] = 0

    return timeWindowMask

def hmmForwardBackward(obs, A_hmm, B_hmm, diagVariance, timeWindowMask, startProb):
    """
    Uses the HMM to infer when each character is likely to have occurred in the data using the forward-backward algorithm. 
    Returns a matrix (T x S) describing the probability of each state occurring at each time step of the neural data sequence 'obs'. 
    See, for example, Chapter 13 of "Bishop, Christopher M. Pattern Recognition and Machine Learning. New York: Springer, 2011." for
    an explanation of the forward-backward algorithm.
    
    Args:
        obs (matrix : T x N): A matrix of neural activity (T observations) recorded during the sentence
        A_hmm (matrix : S x S): The state transition matrix, where entry (i,j) describes the probability of going from state i to j
        B_hmm (matrix : S x N): The emission distributions, where each column i defines the mean firing rates for state i
        diagVariance (vector : N x 1): The variance of each electrode's firing rate (the diagonal of the multivariate covariance)
        timeWindowMask (matrix : T x S): A time window mask that is multiplied against the HMM observation probabilities 
            during forced alignment. If it's zero for a given state and time step, then that state cannot appear at that time step.

    Returns:
        pStates (matrix : T x S): A matrix describing the probability of each HMM state occurring at each moment in time
    """
    
    #number of states in the HMM
    numStates = A_hmm.shape[0] 

    #number of time bins in the sentence plus one
    L = obs.shape[0] + 1 

    #Initialize the forward / backward probability vectors (fs, bs) and scaling factor (s).
    #fs = P(Xt | O1, ..., Ot)
    #bs ∝ P(Ot+1, ..., OT | Xt)
    #fs and bs are combined to compute P(Xt | O1, ... OT) ∝ P(Xt | O1, ..., Ot)P(Ot+1, ..., OT | Xt)

    fs = np.zeros([numStates, L])
    fs[:, 0] = startProb

    bs = np.ones([numStates, L])

    s = np.zeros([L])
    s[0] = 1

    #forward pass
    for count in range(1, L):
        #multivariate normal observation probabilities P(Ot | Xt)
        squaredDiffTerm = -np.square(B_hmm-obs[count-1,:])/(2*diagVariance)
        gaussianEmissionProb = np.exp(np.sum(squaredDiffTerm, axis=1))

        #recursive computation of fs
        fs[:,count] = gaussianEmissionProb * np.matmul(np.transpose(A_hmm), fs[:,count-1]) * timeWindowMask[count-1,:]

        #scaling
        s[count] =  np.sum(fs[:,count])
        fs[:,count] =  fs[:,count]/s[count]   

    #backward pass
    for count in range(L-2,-1,-1):
        #multivariate normal observation probabilities P(Ot+1 | Xt+1)
        squaredDiffTerm = -np.square(B_hmm-obs[count,:])/(2*diagVariance)
        gaussianEmissionProb = np.exp(np.sum(squaredDiffTerm, axis=1))

        #recursive computation of bs
        bs[:,count] = np.matmul(A_hmm, bs[:,count+1] * gaussianEmissionProb * timeWindowMask[count,:])

        #scaling
        bs[:,count] = bs[:,count]/s[count+1]

    #final probabilities
    pSeq = np.sum(np.log(s))
    pStates = fs*bs;

    #get rid of the first column
    pStates = pStates[:, 1:]
    
    return pStates

def hmmViterbi(obs, A_hmm, B_hmm, diagVariance, timeWindowMask, startProb):
    """
    Uses the HMM to infer when each character is likely to have occurred in the data using the Viterbi algorithm. 
    Returns a vector (T x 1) describing the most likely sequence of HMM states given the neural data in 'obs'.
    
    Args:
        obs (matrix : T x N): A matrix of neural activity (T observations) recorded during the sentence
        A_hmm (matrix : S x S): The state transition matrix, where entry (i,j) describes the probability of going from state i to j
        B_hmm (matrix : S x N): The emission distributions, where each column i defines the mean firing rates for state i
        diagVariance (vector : N x 1): The variance of each electrode's firing rate (the diagonal of the multivariate covariance)
        timeWindowMask (matrix : T x S): A time window mask that is multiplied against the HMM observation probabilities 
            during forced alignment. If it's zero for a given state and time step, then that state cannot appear at that time step.
        startProb (vectors : S x 1): Defines the starting probability of each state

    Returns:
        viterbiStates (vector : T x 1): A vector describing the most likely sequence of HMM states given the neural data in 'obs'.
    """

    numStates = A_hmm.shape[0]
    L = obs.shape[0]
    
    #working with log probabilities to avoid numerical issues
    #ignore divide by zero errors since log(0)=-inf is behavior that we want here
    with np.errstate(divide='ignore'):
        logTR = np.log(A_hmm) 

    #pTR stores, for each state S and time step T, the previous state in the most likely path ending in S at time T.
    #This can be used to backtrace the most likely path, beginning with the most likely state on the final time step. 
    pTR = np.zeros([numStates,L])

    #v stores, for each state, the probability of the most likely path that ends in that state at the curent time step.
    with np.errstate(divide='ignore'):
        v = np.log(startProb[:,np.newaxis])

    #loop through each time step, updating pTR and v one step at a time
    for count in range(L):
        #multivariate normal observation probabilities for this time step P(Ot | Xt)
        squaredDiffTerm = -np.square(B_hmm-obs[count,:])/(2*diagVariance)
        gaussianEmissionProb = np.sum(squaredDiffTerm, axis=1, keepdims=True)

        #recursively update v; for each state, find the best way to get there from the previous time step
        #and keep track of it in pTR
        tmpV = v + logTR
        maxIdx = np.argmax(tmpV, axis=0)
        maxVal = np.take_along_axis(tmpV, np.expand_dims(maxIdx, axis=0), axis=0)

        with np.errstate(divide='ignore'):
            v = gaussianEmissionProb + np.transpose(maxVal) + np.log(timeWindowMask[count,:,np.newaxis])
            
        pTR[:,count] = maxIdx

    #decide which of the final states is most probable
    finalState = np.argmax(v)
    logP = v[finalState]

    #Now back trace through pTR to get the most likely state path
    viterbiStates = np.zeros([L]).astype(np.int32)
    viterbiStates[-1] = finalState
    for count in range(L-2,0,-1):
        viterbiStates[count] = pTR[viterbiStates[count+1], count+1]
        
    return viterbiStates
        
def refineCharacterStartTimes(obs, sentence, templates, letterStarts, letterStretches):
    """
    Refines the start times and stretch factors of each character by shifting them around a bit until
    they lie on correlation hotspots with the neural data. 
    
    Args:
        obs (matrix : T x N): A matrix of neural activity (T observations) recorded during the sentence
        sentence (str): The sentence to be labeled
        templates (dict): A dictionary of matrices defining the spatiotemporal pattern of neural activity for each character,
                  indexed by that character's string representation.
        letterStarts (vector : C x 1) A vector describing each character's start time.
        letterStretches (vector : C x 1): A vector describing how each character is contracted/dilated relative to its
                                          template duration. Stretch factors greater than 1 indicate dilation. 
    Returns:
        letterStarts (vector : C x 1) A vector of refined character start times.
        letterStretches (vector : C x 1): A vector of refined character stretch factors.
    """
        
    #Refine each character one at a time, proceeding serially through the sentence
    for c in range(len(sentence)):
        #generate a list of potential start times and stretch factors for this character
        possibleStart = np.arange(letterStarts[c]-50, letterStarts[c]+55, 5).astype(np.int32)
        possibleStart = possibleStart[possibleStart>=0]
        possibleStretch = np.linspace(0.4, 1.5, 15)

        #don't allow start times that are too close to the previous character
        if c>0:
            possibleStart = possibleStart[possibleStart>=(letterStarts[c-1]+20)]

        template = templates[sentence[c]]
        corrHeatmap = np.zeros([len(possibleStretch), len(possibleStart)])
        corrHeatmap[:] = -np.inf

        #compute a correlation heatmap by correlating the template to the data at each stretch factor and start location
        for (stretch, stretchCount) in zip(possibleStretch, range(len(possibleStretch))):
            newX = np.linspace(0,1,int(np.round(template.shape[0]*stretch)))
            stretchedTemplate = np.zeros([len(newX), template.shape[1]])
            for colIdx in range(template.shape[1]):
                stretchedTemplate[:,colIdx] = np.interp(newX, np.linspace(0,1,template.shape[0]), template[:,colIdx])

            for (startIdx, startCount) in zip(possibleStart, range(len(possibleStart))):
                #don't evaluate possibilities that intersect the previous template
                if c>0:
                    prevTemplateEnd = letterStarts[c-1] + letterStretches[c-1]*templates[sentence[c-1]].shape[0]
                    if startIdx < (prevTemplateEnd-10):
                        continue

                #don't evaluate possibilities that intersect the next template
                if c<len(sentence)-1:
                    thisTemplateEnd = startIdx + stretchedTemplate.shape[0]
                    if thisTemplateEnd > letterStarts[c+1]+10:
                        continue

                #don't evaluate possibilities that lie outside of the data range
                stepIdx = np.arange(startIdx, startIdx + stretchedTemplate.shape[0]).astype(np.int32)
                if stepIdx[-1]>=(obs.shape[0]):
                    continue

                #compute correlation between this template and the data
                msDat = obs[stepIdx,:] - np.mean(obs[stepIdx,:], axis=0, keepdims=True)
                msST = stretchedTemplate - np.mean(stretchedTemplate, axis=0, keepdims=True)

                normDat = np.sqrt(np.sum(np.square(msDat), axis=0))
                normST = np.sqrt(np.sum(np.square(msST), axis=0))

                #sometimes the denominator is zero, which will produce nans. np.nanmean then ignores these
                with np.errstate(invalid='ignore'):
                    corrCoeff = np.sum(msDat * msST, axis=0) / (normDat * normST)
                    
                corrHeatmap[stretchCount, startCount] = np.nanmean(corrCoeff)

        #select character stretch factors and start times based on the heatmap hotspot
        maxIdx = np.argmax(corrHeatmap)
        maxIdx = np.unravel_index(maxIdx, corrHeatmap.shape)

        letterStretches[c] = possibleStretch[maxIdx[0]]
        letterStarts[c] = possibleStart[maxIdx[1]]
        
    return letterStarts, letterStretches

def initializeCharacterTemplates(twCubes, charDef):
    """
    Initializes the neural activity templates for each character using time-warped data as input.
    
    Args:
        twCubes (dict): The .mat file containing time-warped data for each character
        charDef (dict): A definition of character names and lengths (see characterDefinitions.py)

    Returns:
        templates (dict): A dictionary of matrices defining the spatiotemporal pattern of neural activity for each character,
                  indexed by that character's string representation.
    """
        
    #Make a template for each character and store it in the 'templates' dictionary
    templates = {}
    for char, charAbbr, thisCharLen in zip(charDef['charList'], charDef['charListAbbr'], charDef['charLen']):
        #Average the time-warped data across trials and smooth it
        neuralCube = twCubes[char].copy()
        neuralCube = np.nanmean(neuralCube, axis=0)
        neuralCube = scipy.ndimage.filters.gaussian_filter1d(neuralCube, 4.0, axis=0)

        #Select the time window of neural activity to use for this template (time step 50 is the 'go' cue)
        neuralCube = neuralCube[59:(59+thisCharLen+1),:]

        #Use PCA to denoise this template by keeping only the top 10 dimensions
        pcaModel = sklearn.decomposition.PCA(n_components=10)
        pcaModel.fit(neuralCube)
        lowRankTemplate = pcaModel.inverse_transform(pcaModel.transform(neuralCube))

        templates[charAbbr] = lowRankTemplate
        
    return templates

def forcedAlignmentLabeling(obs, sentence, templates):
    """
    Labels the neural data in 'obs' using a forced-alignment HMM. Returns the estimated start time and duration of 
    each character. 
    
    Args:
        obs (matrix : T x N): A matrix of neural activity (T observations) recorded during the sentence
        sentence (str): The sentence to be labeled
        templates (dict): A dictionary of matrices defining the spatiotemporal pattern of neural activity for each character,
                  indexed by that character's string representation.
    Returns:
        letterStarts (vector : C x 1) A vector of character start times.
        letterDurations (vector : C x 1): A vector of character durations.
    """
        
    #initialize the HMM parameters
    hmmBinSize = 5
    blankProb = 0.1
    A_hmm, B_hmm, diagVariance, stateLabels, stateLabelsSeq = makeForcedAlignmentHMM(templates, sentence, hmmBinSize, blankProb)
    
    #bin the neural data and append a termination symbol
    nHMMSizedBins = np.floor(obs.shape[0]/hmmBinSize).astype(np.int32)
    obsBinned = np.zeros([nHMMSizedBins, obs.shape[1]])
    binIdx = np.arange(0, hmmBinSize).astype(np.int32)
    for x in range(nHMMSizedBins):
        obsBinned[x,:] = np.mean(obs[binIdx,:], axis=0)
        binIdx += hmmBinSize

    #make a time window mask to prevent pathological solutions
    timeWindowMask = makeTimeWindowMask(stateLabelsSeq, obsBinned.shape[0])
    
    #define starting probabilities
    startProb = np.zeros([A_hmm.shape[0]])
    startProb[0] = blankProb
    startProb[1] = 1 - startProb[0]
    
    #find the Viterbi path
    viterbiStates = hmmViterbi(obsBinned, A_hmm, B_hmm, diagVariance, timeWindowMask, startProb)
    
    #get character start times and character stretch factors from the Viterbi path
    labeledStates = stateLabelsSeq[viterbiStates]

    letterStarts = np.zeros([len(sentence),1])
    letterStretches = np.zeros([len(sentence),1])
    for x in range(len(sentence)):
        thisChar = np.argwhere(labeledStates[:,0]==x)
        letterStarts[x] = (thisChar[0]+1)*hmmBinSize-1
        letterStretches[x] = (len(thisChar)*hmmBinSize)/templates[sentence[x]].shape[0]
    
    #refine the character start times
    letterStarts, letterStretches = refineCharacterStartTimes(obs, sentence, templates, letterStarts, letterStretches)
    
    #return character start times and durations
    letterDurations = np.zeros(letterStretches.shape)
    for x in range(len(sentence)):
        letterDurations[x] = letterStretches[x]*templates[sentence[x]].shape[0]
        
    #return HMM-identified time windows of 'blank' states that can be used to generate synthetic pauses later
    paddedStates = np.concatenate([np.array([0]), labeledStates[:,0], np.array([0])])

    blankStart = np.argwhere(np.logical_and(paddedStates[1:]==-1, paddedStates[0:-1]!=-1))
    blankEnd = np.argwhere(np.logical_and(paddedStates[0:-1]==-1, paddedStates[1:]!=-1))-1
    blankWindows = []

    for b in range(len(blankStart)):
        loopIdx = np.arange((blankStart[b,0]+1)*5-1, (blankEnd[b,0]+2)*5-2).astype(np.int32)
        blankWindows.append(loopIdx)

    return letterStarts, letterDurations, blankWindows

    