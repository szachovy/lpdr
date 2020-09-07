# License Plate Detector & Recognizer (lpdr)

:heavy_check_mark: Low code

:heavy_check_mark: Scalable

:heavy_check_mark: Easy to modify

:heavy_check_mark: Independable

License plate extractor with optical character recognition.


#### :arrow_forward: If you want to understand all the underlying processes please read my <paper> on it. :arrow_backward:

## :rocket: How it works
![diagram](https://github.com/szachovy/lpdr/blob/master/test_images/diagram.png)

## :rocket: Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install lpdr.

```bash
pip install lpdr
```
**or**

_Recommended_: Clone this repo for direct lpdr usage. 

```bash
git clone https://github.com/szachovy/lpdr.git
```

## :rocket: Usage
### Case 1: LPDR as a module
```python
import lpdr
```
#### If you only want to extract license plate from an image.
```python
lpdr.LPD().Y # returns numpy.ndarray
```
#### If you want to get license plate numbers from an image.
```python
lpdr.LPR().output # returns string
```
### Case 2 _Recommended_: LPDR as a microservice

```bash
# in main directory
python lpdr/lpdr.py [-h] [-c CAPTURE] [-i IMAGE_PATH] # returns stdout/stderr
```
where:

  - **-h**, **--help**            _show this help message and exit_
  - **-c CAPTURE**, **--capture CAPTURE** _Captures image from camera after run (0/1 or True/False), **False** by default_
  - **-i IMAGE_PATH**, **--image_path IMAGE_PATH** _Defines input/output path for image (relative or absolute path to the image) **'image.png'** or **'image.jpg'** by default_


Example of CLI usage:
```bash
python lpdr/lpdr.py -c True -i image.png
# captures the picture from a given
# optical source and saves it as image.png for lpdr program execution operations.
```

**Why it is better to use recommended choices?**

You may need to personalize the settings for more specified tasks, as a python package you would first need to find the package path.

Here are the options available, you can change them as you wish.:
- Add tesseract utility for Windows (absolute path to tesseract.exe), it sometimes causes the problems on Windows machines if you do not add it.
- Defaults for argparser (capture / image_path)
- Input camera source
- Width of the image to be captured
- Height of the image to be captured
- Wpod-net path (or different one if you have .h5 and .json files after training)
- IOU threshold
- Confidence if the detected object is a license plate
- Alpha normalization parameter
- Confidence with plate sizing parameters
- Loss function
- OEM
- PSM

## :rocket: Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update the tests as appropriate.

## :rocket: References
[License Plate Detection ECCV 2018 paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/Sergio_Silva_License_Plate_Detection_ECCV_2018_paper.pdf) _Sergio Montazzolli Silva, Claudio Rosito Jung_

[ALPR for unconstrained scenarious](https://github.com/sergiomsilva/alpr-unconstrained)

[Optical Character Recognition by Open source OCR Tool Tesseract: A Case Study](https://www.researchgate.net/profile/Chirag_Patel27/publication/235956427_Optical_Character_Recognition_by_Open_source_OCR_Tool_Tesseract_A_Case_Study/links/00463516fa43a64739000000.pdf) _Chirag Patel, Atul Patel, PhD, Dharmendra Patel_

[An Overview of the Tesseract OCR Engine](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/33418.pdf) _Ray Smith, Google Inc._

[Wrapper for Google's Tesseract-OCR Engine.](https://github.com/madmaze/pytesseract)

## :rocket: Author
- Wiktor Jakub Maj

## :rocket: License
[MIT](https://opensource.org/licenses/MIT)
