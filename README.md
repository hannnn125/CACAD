# Conversational AI for Child Abuse Detection Through Multistage Counseling
---

This repository contains the official implement of "[Conversational AI for Child Abuse Detection Through Multistage Counseling: Model Development and Validation Study](https://www.jmir.org/2026/1/e86536)"
![Overview](./assets/fig2_main.png)

> Hyun-Young Moon<sup>1*</sup>, Youn-Gyu Jin<sup>1*</sup>, Yoon-Ju Kim<sup>1</sup>, Gwang-Cheol Lee<sup>1</sup>, Hyeon-Taek Oh<sup>2</sup>, Hyun A Kim<sup>2</sup>, Dinara Aliyeva<sup>3</sup>, Hyunjoo Na<sup>4</sup>, Kang-Min Kim<sup>5†</sup>
><sup>1</sup>Department of Artificial Intelligence, The Catholic University of Korea, Bucheon, Republic of South Korea
><sup>2</sup>Department of Psychology, The Catholic University of Korea, Bucheon, Republic of South Korea
><sup>3</sup>Department of Computer Science, College of Arts & Sciences, University of North Carolina at Chapel Hill, Chapel Hill, NC, United States
><sup>4</sup>College of Nursing, The Catholic University of Korea, Seoul, Republic of South Korea
><sup>5</sup>Department of Software Convergence, Kyung Hee University, Yongin, Gyeonggi, Republic of South Korea
>*: Equal Contribution, †: Corresponding Author

This repository includes:
-   
-   
-   

---
## Table of Contents
---
1. [Project Structure](#project-structure) 
2. CACAD Setup 

---
## Project Structure 
'''plaintext
├── app/                         
│   ├── CACAD_32B.py
│   ├── CACAD_Gradio.py
│   ├── NQCP.py
│   └── test.py
├── assets/                       
│   ├── fig2_main.png
│   └── fig4_overview copy.png
├── configs/
│   └── base_config.yaml
├── conversations/                
├── data/
│   ├── raw/                      
│   └── processed/
│       ├── labeled_dataset/     
│       ├── finetuning_dataset/
│       └── offensive_dataset/    
├── prompts/
│   ├── cluster_details/          
│   └── counseling/               
├── shells/                       
├── src/
│   ├── abuse_detection/
│   │   ├── MLC/                  
│   │   └── uncertainty/
│   ├── counseling/
│   │   ├── NQCP/                 
│   │   └── offensive_question/
│   └── processing/
│       ├── Clustering/           
│       ├── gen_ft_dataset.py
│       └── preprocess_raw.py
├── .gitignore
└── README.md
'''
---

### Installation

### Dataset


## Citation
---
\```bibtex
@article{moon-etal-2026-cacad,
    title = "Conversational AI for Child Abuse Detection Through Multistage Counseling: Model Development and Validation Study",
    author = "Moon, Hyun-Young and
      Jin, Youn-Gyu and
      Kim, Yoon-Ju and
      Lee, Gwang-Cheol and
      Oh, Hyeon-Taek and
      Kim, Hyun A and
      Aliyeva, Dinara and
      Na, Hyunjoo and
      Kim, Kang-Min",
    journal = "Journal of Medical Internet Research",
    volume = "28",
    pages = "e86536",
    year = "2026",
    doi = "10.2196/86536",
    url = "https://doi.org/10.2196/86536",
}
\```