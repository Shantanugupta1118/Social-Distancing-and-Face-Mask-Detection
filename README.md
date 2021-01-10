# Social Distancing and Face Mask Detection :mask:
[![](https://img.shields.io/badge/Python-%3E%3D3.8-red?style=for-the-badge&logo=python)](https://www.python.org/)   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](https://img.shields.io/badge/yolo%20-v3-yellowgreen?style=for-the-badge&logo=Yolo)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ![](https://img.shields.io/badge/TensorFlow-v2.4.0-blue?style=for-the-badge&logo=TensorFlow) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  

![covid alert!](https://neuralet.com/wp-content/uploads/2020/10/cover2.jpg)
<center><font style size="3" color="red">C O V I D - 1 9   &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;   A L E R T!</font></center>

> ## About-- 
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; ***Social Distancing and Face Mask Detection*** Platform utilizes Artificial Network to perceive if a person walk with maintain social distance and does/doesn’t wear a mask as well. The application can be associated with any current or new IP cameras to identify individuals maintaining social distance with/without a mask. :mask: 
</br>
</br>
> ## System Requirement -- :desktop_computer:

 -  **SOFTWARE--**
	 &nbsp; &nbsp; &nbsp;   
	 * Software: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Anaconda + Python 3.x (3.8 or earlier) 
	 * Editor: &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; VS Code/ PyCharm/ Sublime/ Spyder </br>
	 * Environment: &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; TensorFlow 
	* GPU Drivers:&nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; Nvidia® 		CUDA® 11.0 requires 450.x or above 
		&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; CUDA® Toolkit (TensorFlow >= 2.4.0) 
		 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;cuDNN SDK 8.0.4 (TensorFlow >= 2.4.0)
* **HARDWARE--**
	* GPU: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Graphics Processor (NVIDIA) ̶min 2GB 	&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<font size="1"> *The current development of TensorFlow supports only GPU computing using NVIDIA toolkits and software* </font> 
	*	Camera: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;CCTV/ Webcam/ Mobile Camera (Sharing Camera) 
	*	Storage Disk (Optional): &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;SSD – Min 400MB/s Read Speed	


 
> ## Installation Process--
 * Download Anaconda Software -- 

||Operating System | Download Link  |
|--|--|--|
|:point_right:|Windows | [click here](https://docs.anaconda.com/anaconda/install/windows/)  |
|:point_right:|Mac|[click here](https://docs.anaconda.com/anaconda/install/mac-os/)
|:point_right:|Linux | [click here](https://docs.anaconda.com/anaconda/install/linux/)
:loudspeaker: During Installation be sure to check to set *Environmental variable path* 

* Create new Environment for the installation of libraries:
	* Open Command Prompt / Anaconda Prompt and type `conda create --name tf_python`  
	you can set any name in place of tf_python to create a new envionment. and after type `y` and enter.
	* Install all required Libraries given in <font color="red">requirement.txt</font> by using command `pip install -r requirement.txt`
	 
> #### Required Libraries--
* ![](https://img.shields.io/badge/TensorFlow-v2.4.0-blue)   &nbsp; &nbsp; &nbsp;  &nbsp;&nbsp;  [TensorFlow](https://pypi.org/project/tensorflow/)
* ![](https://img.shields.io/badge/TensorFlow--GPU-v2.4.0-blue) &nbsp;   &nbsp;[TensorFlow-GPU](https://pypi.org/project/tensorflow-gpu/)
* ![](https://img.shields.io/badge/python-v3.7-blue) &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; [Python](https://www.python.org/downloads/)
*  ![](https://img.shields.io/badge/SciKit%20Learn-v0.24.0-blue) &nbsp; &nbsp; &nbsp; [Scikit Learn](https://pypi.org/project/scikit-learn/)
* ![](https://img.shields.io/badge/Open%20CV-v4.4.0.46-blue)    &nbsp;   &nbsp; &nbsp; &nbsp; [Computer Vision](https://pypi.org/project/opencv-python/)
* ![](https://img.shields.io/badge/SciPy-v1.6.0-blue) &nbsp;   &nbsp; &nbsp;  &nbsp;  &nbsp;  &nbsp;&nbsp;  &nbsp;  &nbsp;[Scientific Python](https://pypi.org/project/scipy/)

 
> ## File Required to Download --
 * **DATASETS :** 
	 &nbsp;  &nbsp;  &nbsp;  &nbsp; Using datasets to train the model for ***Face Mask Detection*** model. To download the dataset -- :point_right:<a href="https://www.kaggle.com/shantanu1118/face-mask-detection-dataset-with-4k-samples"> *Click here* </a>  &nbsp;:point_left: (Dataset with 4,000 Images Sampels) <font size="2"> </br> 
	 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; :star2:File contain 2 Sub-Folder i.e. With_mask & Without_mask (each folder contain 2k samples of images)). 


 *	 **Yolo Weights  (V3) -- Pre-Trained model:**   
 &nbsp; &nbsp; &nbsp; &nbsp; YOLO *(You Only Live Once)*, the pre-trained weights of the neural network are stored in `yolov3.weights`
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Download the Weight File :point_right: <a href="">*Click here* </a> :point_left: 

> ## File Structure 
Set all downloaded files to their respective folders/path as given in Folder Structure Diagram.

> ## RUN the Main Module--
*	Using Command Prompt or Anaconda Prompt:
	*  To activate environment:--&nbsp; &nbsp; &nbsp; `conda activate tf_python`
	* Run main module:-- `python main.py`
	

> ## Outputs--

> ## Contribute:
&nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;<b>:fire: Contributions are always welcome!</b>
# Drop a :star: if you like this Repository.. :smile: 
	
 
[![enjoy][enjoy-image]][linkedin-url] [![status][issue-image]][issue-url]

[enjoy-image]: https://img.shields.io/badge/Enjoy%20this%3F-Say%20Thanks!-yellow
[linkedin-url]: https://www.linkedin.com/in/shang1118/
[issue-image]: https://img.shields.io/badge/Any%20Issues%3F-Track%20Issue-red
[issue-url]: https://github.com/Shantanugupta1118/Social-Distancing-and-Face-Mask-Detection/issues
:point_right::point_right::point_right: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[![](https://img.shields.io/badge/Shantanu%20-Gupta-orange?style=for-the-badge&logo=Coder)](https://shantanug.000webhostapp.com/)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;:point_left::point_left::point_left:
