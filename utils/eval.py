import numpy as np

def mean_squared_error_depth(predicted: np.ndarray, gt: np.ndarray)->float:
    '''
    Takes two 2D np array and calculates the pixelwise MSE error between them (normalized by max value)
    args:
        predicted: predicted depth map
        gt: ground truth depth map
    
    return:
        mse: mean squared error pixelwise

    '''
    assert predicted.shape == gt.shape 
    #normalize both
    gt = gt/gt.max()
    mask = np.array(gt, dtype=bool).astype(int)

    predicted = (predicted*mask) #element wise multiply the mask
    predicted = predicted/predicted.max()

    mse = ((gt-predicted)**2).mean()
    return float(mse)


if __name__=='__main__':
    a = np.random.rand(500,500)
    b = np.random.rand(500,500)

    print(mean_squared_error_depth(a,b))
