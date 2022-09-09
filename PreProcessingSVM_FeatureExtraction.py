import os 
import json 

import numpy as np 
import cv2 
import matplotlib.pyplot as plt 
from skimage.feature import greycomatrix, greycoprops
from skimage import io, color, img_as_ubyte
DATASET_PATH = "SVM_Tumor" #veri seti dosyası
JSON_PATH = "SVM_Dataset_Feature_Hazir.json" #kaydedeceğimiz dosya

#%% Statical Features Hesaplamada kullanılan GLCM fonksiyonlar 
def contrast_feature(matrix_coocurrence):
	contrast = greycoprops(matrix_coocurrence, 'contrast')
	return "Contrast = ", contrast

def dissimilarity_feature(matrix_coocurrence):
	dissimilarity = greycoprops(matrix_coocurrence, 'dissimilarity')	
	return "Dissimilarity = ", dissimilarity

def homogeneity_feature(matrix_coocurrence):
	homogeneity = greycoprops(matrix_coocurrence, 'homogeneity')
	return "Homogeneity = ", homogeneity

def energy_feature(matrix_coocurrence):
	energy = greycoprops(matrix_coocurrence, 'energy')
	return "Energy = ", energy

def correlation_feature(matrix_coocurrence):
	correlation = greycoprops(matrix_coocurrence, 'correlation')
	return "Correlation = ", correlation

def asm_feature(matrix_coocurrence):
	asm = greycoprops(matrix_coocurrence, 'ASM')
	return "ASM = ", asm
def save_feature(dataset_path, json_path):
 #%%   
    #dictionary to store data 
    data = {
        "mapping":  [], #classical, blues
        "Features":     [], #
        "labels":   [] #class'ların labelları
        }    
    
    
    #loop through all the genres
    for FB, (dirpath, dirnames, filenames) in enumerate( os.walk(dataset_path)):
        pass
    
        #ensure that we're not at the root level 
        if dirpath is not dataset_path:
            pass
    
            #save the semantic label. mapping'e labelları alıyoruz
            dirpath_components = dirpath.split("/") #genre/blues => ["genre", "blues"]
            semantic_label = dirpath_components[-1] # => ["blues]
            data["mapping"].append(semantic_label)
            print("\nProcessing {}".format(semantic_label))
            
            
            # process files for a specific genre 
            for f in filenames:
                #load 
                file_path = os.path.join(dirpath, f)
                #read image
                img = cv2.imread(file_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                #%% ilk 5 GaborKernel oluşturuyolar ve tek tek imgeye uygulanıyor
                ksize = 15 #kernel size
                sigma = 5 #standard deviation of the gaussian function. Diğer kombinasyonalr için bunlada oynayabilirsin.
                # bi 5 yap. bi 20 mesela
                lamda = 1*np.pi/4 # dalga boyu # 1/lamda frekansı temsil ediyor 
                gamma = 0.1 # aspect ratio
                phi = 0 #faz # faz ile oynarak diğer kombinasyonları elde et. fazı 0.5 yap

                GaborKernels =  list()
                FilteredImage = list()
                for i in range(5):
                    theta = 1*np.pi/2 * i #orientation 0 45 90 135 180 
                    myKernel = cv2.getGaborKernel((ksize,ksize), sigma, theta, lamda, gamma, phi, ktype = cv2.CV_32F)
                    GaborKernels.append(myKernel);

                for i in range(5):
                    fimg = cv2.filter2D(img, cv2.CV_8UC3,GaborKernels[i])
                    plt.imshow(fimg)
                    FilteredImage.append(fimg)

                #%% Son 5 GaborKernel oluşturuluyor ve tek tek imgeye uygulanıyor 
                ksize = 15 #kernel size
                sigma = 5 #standard deviation of the gaussian function. Diğer kombinasyonalr için bunlada oynayabilirsin.
                # bi 5 yap. bi 20 mesela
                lamda = 1*np.pi/16 # dalga boyu # 1/lamda frekansı temsil ediyor 
                gamma = 0.1 # aspect ratio
                phi = 0 #faz # faz ile oynarak diğer kombinasyonları elde et. fazı 0.5 yap


                for i in range(5):
                    theta = 1*np.pi/2 * i #orientation 0 45 90 135 180 
                    myKernel = cv2.getGaborKernel((ksize,ksize), sigma, theta, lamda, gamma, phi, ktype = cv2.CV_32F)
                    GaborKernels.append(myKernel);
                for i in range(5):
                    ffimg = cv2.filter2D(img, cv2.CV_8UC3,GaborKernels[i+5])
                    plt.imshow(ffimg)
                    FilteredImage.append(ffimg)

                #%%
                # Gabor Kernels değişkeninin içine 10 kombinasyonda 10 gabor filtresi üretildi 
                # FilteredImage değişkeninin içine örnek imgeye filtrelenmiş imgeye uygulanmış halleri var.



                #%% Contrast Hesaplama
                FilteredImageContrasts = list()
                for i in range(10):
    
                    gray = color.rgb2gray(FilteredImage[i])
                    image = img_as_ubyte(gray)
                    bins = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255]) #16-bit
                    inds = np.digitize(image, bins)
                    max_value = inds.max()+1
                    matrix_coocurrence = greycomatrix(inds, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=max_value, normed=False, symmetric=False)
                    Contrast_Temp = 0 ;
                    Contrast_Toplam= 0 ;
                    contrast = contrast_feature(matrix_coocurrence);
                    TempContrastArray = np.asarray(contrast) # tupple to array
                    TempContrastArray = TempContrastArray[1]
                    for i in range(4):
                        Contrast_Temp = TempContrastArray[0,i]
                        Contrast_Toplam += Contrast_Temp
        
                    contrast = Contrast_Toplam / 4
                    FilteredImageContrasts.append(contrast)

#%% Dissimilarity Hesaplama
                FilteredImageDissimilarity = list()
                for i in range(10):
                    gray = color.rgb2gray(FilteredImage[i])
                    image = img_as_ubyte(gray)
                    bins = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255]) #16-bit
                    inds = np.digitize(image, bins)
                    max_value = inds.max()+1
                    matrix_coocurrence = greycomatrix(inds, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=max_value, normed=False, symmetric=False)
                    
                    Dissimilarity_Temp = 0 
                    Dissimilarity_Toplam = 0 
                    dissimilarity = dissimilarity_feature(matrix_coocurrence);
                    TempDissimilarityArray = np.asarray(dissimilarity) # tupple to array
                    TempDissimilarityArray = TempDissimilarityArray[1]
                    for i in range(4):
                        Dissimilarity_Temp = TempDissimilarityArray[0,i]
                        Dissimilarity_Toplam += Dissimilarity_Temp
                    dissimilarity = Dissimilarity_Toplam / 4
                    FilteredImageDissimilarity.append(dissimilarity)
    
#%% Homogeneity Hesaplama
                FilteredImageHomogeneity = list()
                for i in range(10):
                    gray = color.rgb2gray(FilteredImage[i])
                    image = img_as_ubyte(gray)
                    bins = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255]) #16-bit
                    inds = np.digitize(image, bins)
                    max_value = inds.max()+1
                    matrix_coocurrence = greycomatrix(inds, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=max_value, normed=False, symmetric=False)
                    
                    Homogeneity_Temp = 0 
                    Homogeneity_Toplam = 0 
                    homogeneity = homogeneity_feature(matrix_coocurrence);
                    TempHomogeneityArray = np.asarray(homogeneity)
                    TempHomogeneityArray = TempHomogeneityArray[1]
                    for i in range(4):
                        Homogeneity_Temp = TempHomogeneityArray[0,i]
                        Homogeneity_Toplam+=  Homogeneity_Temp
                    homogeneity = Homogeneity_Toplam /4
                    FilteredImageHomogeneity.append(homogeneity)
    
#%% Energy Hesaplama 
                FilteredImageEnergy = list()
                for i in range(10):
                    gray = color.rgb2gray(FilteredImage[i])
                    image = img_as_ubyte(gray)
                    bins = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255]) #16-bit
                    inds = np.digitize(image, bins)
                    max_value = inds.max()+1
                    matrix_coocurrence = greycomatrix(inds, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=max_value, normed=False, symmetric=False)
   
    
                    Energy_Temp = 0 
                    Energy_Toplam = 0 
                    energy = energy_feature(matrix_coocurrence);
                    TempEnergyArray = np.asarray(energy)
                    TempEnergyArray = TempEnergyArray[1]
                    for  i in range(4):
                        Energy_Temp = TempEnergyArray[0,i]
                        Energy_Toplam += Energy_Temp
                    energy = Energy_Toplam / 4
                    FilteredImageEnergy.append(energy)

#%% Korelasyon Hesaplama 
                FilteredImageCorrelation = list()
                for i in range(10):
                    gray = color.rgb2gray(FilteredImage[i])
                    image = img_as_ubyte(gray)
                    bins = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255]) #16-bit
                    inds = np.digitize(image, bins)
                    max_value = inds.max()+1
                    matrix_coocurrence = greycomatrix(inds, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=max_value, normed=False, symmetric=False)
                    
                    Correlation_Temp = 0 
                    Correlation_Toplam = 0 
                    correlation = correlation_feature(matrix_coocurrence);
                    TempCorrelationArray = np.asarray(correlation)
                    TempCorrelationArray = TempCorrelationArray[1]
                    for i in range(4):
                        Correlation_Temp = TempCorrelationArray[0,i]
                        Correlation_Toplam += Correlation_Temp
                    correlation = Correlation_Toplam/4
                    FilteredImageCorrelation.append(correlation)
    
                AllFeatures = list()
                for m in range(10):
                    
                    AllFeatures.append(FilteredImageCorrelation[m])
                    AllFeatures.append(FilteredImageContrasts[m])
                    AllFeatures.append(FilteredImageDissimilarity[m])
                    AllFeatures.append(FilteredImageEnergy[m])
                    AllFeatures.append(FilteredImageHomogeneity[m])
                data["Features"].append(AllFeatures)
                data["labels"].append(FB-1)
                print("{}".format(file_path))
                        
                        
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent = 4)
        
if __name__ == "__main__":
    save_feature(DATASET_PATH, JSON_PATH)
    
                    