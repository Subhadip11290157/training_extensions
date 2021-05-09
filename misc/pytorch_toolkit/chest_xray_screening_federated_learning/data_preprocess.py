import os, csv, argparse, math
import numpy as np
from skimage import io
from skimage.transform import resize

def zero_pad(f):
    
    if f.ndim==3:
        f=(f[:,:,0]+f[p:,:,1]+f[:,:,2])/3
        f=np.squeeze(f)
    
    szx=f.shape[0]
    szy=f.shape[1]
    
    if szx<szy:
        df=szy-szx
        df1=math.floor(df/2)
        df2=math.ceil(df/2)
        
        pd1=np.zeros((df1, szy))
        pd2=np.zeros((df2, szy))
        
        
        f=np.concatenate((pd1, f, pd2), axis=0)
        
    elif szy<szx:
        df=szx-szy
        df1=math.floor(df/2)
        df2=math.ceil(df/2)
        
        pd1=np.zeros((szx, df1))
        pd2=np.zeros((szx, df2))
        
        f=np.concatenate((pd1, f, pd2), axis=1)
        
    return f

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dpath', default='', type=str, help='CheXpert data path')
    args = parser.parse_args()
    
    img_path = args.dpath + 'CheXpert-v1.0/' 
    save_path='./Data/Train_reshape/'
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    
    with open(img_path+'train.csv') as csv_file:
        csv_reader=csv.reader(csv_file, delimiter=',')
        header = next(csv_reader)
        if header != None:
            for row in csv_reader:
                
                img_nm=row[0]
                f=io.imread(args.dpath+img_nm)
                dt=f.dtype
                
                print(f.shape)
                f=f.astype(np.float64)
                f=zero_pad(f)
                
                print(str(np.max(f))+'\t'+str(f.dtype)+'\t'+str(f.shape))
            
                # resize to 448 by 448
                f=resize(f, (448, 448), anti_aliasing=True)
                f=f.astype(dt)
                print(str(np.max(f))+'\t'+str(f.dtype)+'\t'+str(f.shape))
                
                img_nm=img_nm.replace("/", "__")
                
                # write image to disk
                io.imsave(save_path+img_nm,f)
                del f
