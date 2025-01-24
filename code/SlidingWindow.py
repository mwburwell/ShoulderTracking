from typing import Annotated, Literal, TypeVar

import numpy as np
import numpy.typing as npt

import matplotlib.pyplot as plt
import math

from sklearn.preprocessing import normalize

from line import Line

DType = TypeVar("DType", bound=np.generic)

LineVector = Annotated[npt.NDArray[DType], Literal[3]]

class SlidingWindow():
    def __init__(self, 
                 img: npt.NDArray, 
                 num_steps: int,  
                 *,
                 median: int = None,
                 medians: list[int] = [],
                 sigma: int = None,
                 sigmii: list[int] = [],
                 constant: int = 10,
                 line: LineVector[np.float64] = np.array([0, 0, 0]),
                 strip: int = 0,
                 horiz: bool= False, 
                 vert: bool= True, 
                 top: bool= False, 
                 bot: bool= False,
                 diagnose: bool= False,
                 diag_num: int= 1
                 ) -> None:
        # parameters
        self.img = img #if horiz or not vert else img.copy()[:, :-strip]
        # print(f'img: {img.shape}')
        self.img_width = self.img.shape[1]
        # print(f'width: {self.img_width}')
        self.img_height = self.img.shape[0]
        # print(f'height: {self.img_height}')
        self.window_length = self.img_height if vert and not horiz else self.img_width
        # print(f'length: {self.window_length}')
        self.num_steps = num_steps
        
        self.window_width = round(self.img_height / (self.num_steps / 2)) if horiz or not vert else round(self.img_width / (self.num_steps / 2))
        # print(f'window_width: {self.window_width}')
        if len(medians) != len(sigmii):
            raise Exception(f"Medians and sigmii need to be same size. <medians: len({len(medians)}> - <sigmii: len({len(sigmii)}))")
        
        self.medians = [median] if median is not None and len(medians) == 0 else medians
        self.sigmii = [sigma] if sigma is not None and len(sigmii) == 0 else sigmii
        # print(f"median: {self.medians}")
        # print(f"sigmii: {self.sigmii}")
        self.normals = np.asarray([
            self.normalDistVector(size=self.window_length, median=self.medians[i], sigma=self.sigmii[i]) 
            for i in range(len(self.medians))]).transpose()
        
        self.segm_line = Line(line)
        self.constant = constant
        
        # booleans
        self.isHoriz = horiz
        self.isVert = vert
        self.isTop = top
        self.isBot = bot
        self.isDiagnosing = diagnose
        self.diag_num = diag_num
        
        self.x_positions, self.y_positions = self.slide()
        self.vector = line
        
        
    def slide(self):
        if self.isDiagnosing:
            self.num_steps= self.diag_num
        
        position_x = []
        position_y = []
        step_length = round(self.window_width / 2)
        for step in range(self.num_steps):
            current_loc = step * step_length
            window = self.getSlide(current_loc)
            
            if self.isHoriz or not self.isVert:
                window = window.transpose()
                
                # break the window if it's current location is crossing the line
                # found in the upper and lower segmentation sliding window
                if self.isTop and current_loc  > self.segm_line.getY(self.img_width / 2) - self.window_width:
                    continue
                elif self.isBot and current_loc < self.segm_line.getY(self.img_width / 2) + self.window_width or current_loc + self.window_width > self.img_height:
                    continue
                    
                
            window_vector = self.computeWindow(window)
            if self.isDiagnosing:
                print(f"After Computing: step {step}")
                print(f"window vector: {window_vector.shape}")
                print(f"Window: {window.shape}")
                print(f"Normals: {self.normals.shape}\n")
                
            max_medians = []
            for i in range(self.normals.shape[1]):
                argument_max = np.argmax(window_vector[:, i])
                window_vector[argument_max, i] = 0
                max_medians.append(argument_max)
                
                # if self.isDiagnosing:
                #     print(i)
                #     print(f"argument max: {argument_max}")
        
            if self.isHoriz or not self.isVert:
                position_x.append(np.asarray(max_medians))
                position_y.append(np.asarray([round(current_loc + (self.window_width / 2)) for i in range(self.normals.shape[1])] ))
            else:
                position_y.append(np.asarray(max_medians))
                position_x.append(round(current_loc + (self.window_width / 2)))

        
        if self.isHoriz or not self.isVert:
            if self.isDiagnosing:
                print(f"Returning: ")
                print(f"x's: {np.asarray(position_x).shape}")
                # print(f"next index: {position_y[0]}")
                print(f"y's: {np.asarray(position_y).shape}")
            
            return np.asarray(position_x), np.asarray(position_y)
        
        return np.asarray(position_x)[np.newaxis].transpose(), np.asarray(position_y)

    
    def getSlide(self, window_location):
        '''
        Slices the image by the defined window sizes at the current location
        '''
        if self.isVert and not self.isHoriz:
            return self.img[:self.window_length, window_location: window_location + self.window_width].copy()
            
        return self.img[window_location: window_location + self.window_width, :self.window_length].copy()


    def computeWindow(self, window):
        
            # v_probabilites = np.ones((v_probabilites.shape[0], 1))
          
        # get the average of each row by vector multiplication
        # with a vector of ones
        kernel = np.ones((window.shape[1], 1), dtype=np.int32)
        avg_intensity_vector = np.matmul(window, kernel) / self.window_width
        
        # find the maximum intensity in the averages and assign a probability
        # for each of the intensities in the average intensity vector
        #
        # there is a constant that I am not sure what it does, but was in
        # the slides
        D_max = np.max(avg_intensity_vector)
        if D_max == 0:
            prob_Di = 0
        else:
            prob_Di = self.constant * (1 - (avg_intensity_vector / D_max) )
        
        
        if self.isDiagnosing:
            print(f"window: {window.shape}")
            print(f"kernel: {kernel.shape}")
            print(f"intensity: {avg_intensity_vector.shape}")
            print(f"prob_di: {prob_Di.shape}")
            print(f"normals: {self.normals.shape}")
        
        
        # multiply the normally distributed probability vector with
        # the intensity average probability vector elementwise
        if self.isHoriz or not self.isVert:
            return np.multiply(self.normals, prob_Di)
        
        
        return  np.multiply(prob_Di, self.normals)

    
    def gaussianProbability(self, y,*, y_hat=210, sigma=1):
        '''
        Uses the gaussian formula to determine the probabability
        of a given "y" value where the median is y_hat and the sigma
        is another value
        '''
        constant = (math.sqrt(2 * math.pi) * sigma)
        exponent = (-1 * (((y - y_hat) ** 2) / (sigma ** 2)))
        probability = (1 / constant) * math.pow(math.e, (exponent / 2))
        
        return probability


    def normalDistVector(self, *, size=472, median=210, sigma=1, horiz=False, vert=True):
        '''
        With the gaussianProbability function this builds a vector
        of given size that has the percentage of likelihood at the
        given possition based on the median and sigma.
        '''
        probs = []
        for i in range(size):
            prob = self.gaussianProbability(i, y_hat= median, sigma=sigma)
            probs.append(prob)
        
        if horiz or not vert:
            return np.asarray(probs)
        
        return np.array(probs)

    def sampling(self, sample_num, sample_title):
        step_length = round(self.window_width / 2)

        # Show single sample size for testing purposes
        test_window_loc = sample_num * step_length
        test_window = self.getSlide(test_window_loc)
        # print(f"top_normals: {top_normals.shape}")
        # vector_normal_probabilites = normalDistVector(size=test_window.shape[1], median=median, sigma=sigma)
        # print(f"Normals: {vector_normal_probabilites.shape}")

        # window_vector = computeWindow(test_window, vector_normal_probabilites)
        vector = [0, 1]
        window_x_values = [test_window_loc, test_window_loc + self.window_width]
        if self.isHoriz or not self.isVert:
            test_window = test_window.transpose()
            
        # print(window.shape)
        kernel = np.ones((self.window_width, 1), dtype=np.int32)
        # print(kernel.shape)
        
        
        avg_intensity_vector = np.matmul(test_window, kernel) / self.window_width
        avg_intens_norm = normalize(avg_intensity_vector, axis=0).ravel()
        
        # normals = normalDistVector(size=window.shape[0], median=median, sigma=sigma)

        ## plot the current slide window
        # add the lines for the slide window
        fig, axes = plt.subplots(1, 3)
        fig.suptitle(sample_title)
        axes[0].set_title("Window Location")
        axes[0].imshow(self.img, cmap="gray")
        if self.isHoriz or not self.isVert:
            axes[0].axhline(test_window_loc, xmin=0, xmax=1)
            axes[0].axhline(test_window_loc + self.window_width, xmin=0, xmax=1)
        else:
            axes[0].axvline(test_window_loc, ymin=0, ymax=1)
            axes[0].axvline(test_window_loc + self.window_width, ymin=0, ymax=1)
        axes[0].plot()
        
        
        axes[1].set_title("Sample")
        axes[1].imshow(test_window, cmap="gray")

        # intensities
        axes[2].set_title("Avg. Intens.")
        axes[2].invert_yaxis()
        axes[2].scatter(avg_intens_norm, range(0, len(avg_intens_norm)))
        
        # normal distribution plot
        # for i in range(normals.shape[0]):
        #     print(i)
        axes[2].plot(self.normals, range(0, len(avg_intensity_vector)), color="red")
        
        plt.show()
    
            