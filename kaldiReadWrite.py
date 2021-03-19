import numpy as np

def writeKaldiProbabilityMatrix(charProb, i, kaldiFile):
    """
    Converts a numpy character probability matrix 'charProb' into a Kaldi matrix file for language model decoding.
    """
    kFile = open(kaldiFile,'w');
    kFile.write(str(i) + ' [\n');
    for t in range(charProb.shape[0]):
        for k in range(charProb.shape[1]):
            kFile.write('%.10g ' % charProb[t,k])
        kFile.write('\n')

    kFile.write(']\n')
    kFile.close()
    
def readKaldiLatticeFile(fileName, fileType):
    """
    Extract the candidate sentences frmo a Kaldi lattice file.
    """
    file = open(fileName, "r")
    allLines = file.readlines()

    sentenceNumber = np.zeros([len(allLines),2])
    content = []
    
    for x in range(len(allLines)):
        splitStr = allLines[x].split(' ')
        
        num = splitStr[0]
        num = num.split('-')
        
        sentenceNumber[x,0] = int(num[0])
        sentenceNumber[x,1] = int(num[1])
        
        if fileType=='numeric':
            content.append(float(splitStr[1]))
        elif fileType=='string':
            del splitStr[0]
            for x in range(len(splitStr)):
                if splitStr[x]=='<space>':
                    splitStr[x] = ' '
                elif splitStr[x]=='\n':
                    splitStr[x] = ''
            
            content.append(''.join(splitStr))
        else:
            print('Wrong file type.')
            raise ValueError
            
    if fileType=='numeric':
        content = np.array(content)
        
    return sentenceNumber, content

def readKaldiAliFile(fileName):
    """
    Extract the inferred state sequences from an alignment file.
    """
    file = open(fileName, "r")
    allLines = file.readlines()

    sentenceNumber = np.zeros([len(allLines),2])
    content = []
    
    for x in range(len(allLines)):
        splitStr = allLines[x].split(' ')
        
        num = splitStr[0]
        num = num.split('-')
        
        sentenceNumber[x,0] = int(num[0])
        sentenceNumber[x,1] = int(num[1])
        
        tmp = np.zeros([len(splitStr)-1])
        for x in range(1, len(splitStr)-1):
            tmp[x] = int(splitStr[x])
            
        content.append(tmp)
        
    content = np.stack(content,axis=0)
        
    return sentenceNumber, content