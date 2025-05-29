import numpy as np

def RANSAC(A, b, nEquations, iterations, threshold):
    """
        calculates the consensus set for the overdetermined linear system of equations Ax = b
        A and b consist of the samples as rows
        
        + A: np.array >> coefficient matrix
        + b: np.array >> right side
        + nEquations: int >> number of equations for one sample
        + iterations: int >> number of iterations
        + threshold: float >> threshold on the norm of the error on the right side

        - consesusA: np.array >> consensus set of the coefficient matrix
        - consensusB: np.array >> consensus set of the right side
        - nOutliers: int >> number of outliers
    """
    
    A = np.asarray(A)
    b = np.asarray(b)
    
    assert A.shape[0] == b.shape[0], "Error: A and b need to have the same amount of rows"
    assert A.shape[0]%nEquations == 0 and b.shape[0]%nEquations == 0, "Error: the number of rows dont match with the number of equations per sample"
    
    
    nVariables = A.shape[1]
    iterationList = []
    
    for i in range(iterations):
        
        # >> random samples can contain duplicates <<
        arr = np.random.randint(0, A.shape[0], size=int(nVariables/nEquations))
        
        # >> for random samples without duplicates use code below <<
        # sample_count = A.shape[0] // nEquations
        # arr = np.random.choice(sample_count, size=int(nVariables / nEquations), replace=False)
        
        arr = arr * nEquations # indices must be a multiple of the number of equations
        
        randomA = np.empty((0, nVariables))
        randomB = np.empty((0,1))
                
        for index in arr:
            clusterA = A[index : index + nEquations,:]
            clusterB = b[index : index + nEquations]
            randomA = np.vstack((randomA, clusterA))
            randomB = np.vstack((randomB, clusterB))
        
        assert randomA.shape[0] == randomA.shape[1], "Error: randomA is not quadratic"
        
        if np.linalg.cond(randomA) > 1e10:
            continue
        
        randomX = np.linalg.solve(randomA,randomB)
        
        error = A.dot(randomX) - b.reshape(-1,1) # error on right side
        residues = []
        
        for j in range(int(error.shape[0]/nEquations)):
            residues.append(np.linalg.norm(error[j : j + nEquations]))
        
        inlierListA = []
        inlierListB = []
            
        for j in range(len(residues)):
            if residues[j] < threshold:
                inlierListA.append(A[j : j + nEquations,:])
                inlierListB.append(b[j : j + nEquations])
        
        iterationList.append((len(inlierListA), inlierListA, inlierListB))
        
    consensusIndex = max(range(len(iterationList)), key=lambda i: iterationList[i][0])
    
    consensusA = np.asarray(iterationList[consensusIndex][1]).reshape(-1,nVariables)
    consensusB = np.asarray(iterationList[consensusIndex][2]).reshape(-1)
    nOutliers = A.shape[0] / nEquations - iterationList[consensusIndex][0]
    
    return consensusA, consensusB, nOutliers