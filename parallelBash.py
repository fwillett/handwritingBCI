from datetime import datetime
import numpy as np
import os
    
def parallelBash(argList, scriptFile, bashFilePrefix, nProcesses):
    """
    A simple utility for making a series of bash scripts that can launch parallel instances of programs.
    We use this to launch multiple kaldi decoders at the same time. 
    
    argList is a list of argument dictionaries that will be passed to the program for each run
    scriptFile is a path to the python script to run
    bashFilePrefix is that name of the bash scripts this function will create
    nProcesses is the number of parallel processes to launch
    """
    nRuns = len(argList)
    nPerProcess = int(np.ceil(nRuns / nProcesses))
    runIdx = 0
    
    #put all tasks belonging to a single process in a bash script that launches them serially
    for processIdx in range(nProcesses):
        fileName = bashFilePrefix + '_' + str(processIdx) + '.sh' 
        file = open(fileName,'w') 
        
        for i in range(nPerProcess):
            if runIdx>=len(argList):
                break
                
            commandStr = scriptFile
            args = argList[runIdx]
            
            for argName in args:
                if argName.endswith('_mainArg'):
                    commandStr = commandStr + ' ' + str(args[argName])
                else:
                    commandStr = commandStr + ' --' + argName + ' ' + str(args[argName])

            commandStr = commandStr + ' \n'
            file.write(commandStr)
            
            runIdx += 1

        file.close()    
        os.chmod(fileName, 0o777)

    #master script that launches all processes at once
    fileName = bashFilePrefix + '_master.sh'
    file = open(fileName,'w') 
    for processIdx in range(nProcesses):
        commandStr = bashFilePrefix + '_' + str(processIdx) + '.sh & \n'
        file.write(commandStr)

    file.close()    
    os.chmod(fileName, 0o777)