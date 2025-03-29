# SCHP Garment-Person Parsing

This notebook uses the Semantic Clothing Parsing (SCHP) model to:
- Parse human and garment images.
- Identify clothing categories (top, bottom, full, etc.).
- Remove matching clothing regions from the person image.
- Overlay a final mask showing the removed area.

### Key Models:
- **LIP model**: For fine-grained garment parsing.
- **PASCAL-Person-Part model**: For anatomical body parts segmentation.

### Applications:
This system can serve as a pre-processing step for virtual try-on, fashion image generation, or body region estimation tasks.

### Assumptions

- The SCHP model is capable of accurately classifying the garment.
- The person in the image is standing upright, with a clear and uncluttered background.
- The garment image also features a clean and unobstructed background.


### Required Model Files

Please download the following pretrained models manually before running the notebook:

- **SCHP (LIP dataset) Model**  
  [Download SCHP Model](https://drive.google.com/file/d/1k4dllHpu0bdx38J7H28rVVLpU-kOHmnH/view?usp=sharing)

- **Pascal-Person-Part Model**  
  [Download Pascal Model](https://drive.google.com/file/d/1E5YwNKW2VOEayK9mWCS3Kpsxf-3z04ZE/view?usp=sharing)

After downloading, upload the `.pth` files into your models folder as follows: -<br>
project/<br>
├── models/<br>
├── networks/<br>
├── img1.jpg<br>
<br>

SCHP Repo: https://github.com/GoGoDuck912/Self-Correction-Human-Parsing
