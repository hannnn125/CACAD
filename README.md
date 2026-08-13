# Conversational AI for Child Abuse Detection Through Multistage Counseling

---

This repository contains the official implement of "[Conversational AI for Child Abuse Detection Through Multistage Counseling: Model Development and Validation Study](https://www.jmir.org/2026/1/e86536)"
![overview](./assets/main.png)

> Hyun-Young Moon<sup>1*</sup>, Youn-Gyu Jin1*, Yoon-Ju Kim1, Gwang-Cheol Lee1, Hyeon-Taek Oh2, Hyun A Kim2, Dinara Aliyeva3, Hyunjoo Na4, Kang-Min Kim5†
> 1Department of Artificial Intelligence, The Catholic University of Korea, Bucheon, Republic of South Korea
> 2Department of Psychology, The Catholic University of Korea, Bucheon, Republic of South Korea
> 3Department of Computer Science, College of Arts & Sciences, University of North Carolina at Chapel Hill, Chapel Hill, NC, United States
> 4College of Nursing, The Catholic University of Korea, Seoul, Republic of South Korea
> 5Department of Software Convergence, Kyung Hee University, Yongin, Gyeonggi, Republic of South Korea
> *: Equal Contribution, †: Corresponding Author

---
## This repository includes:
- CACAD Setup
- Model Training
- Chatbot

---
## Table of Contents
---
1. [Project Structure](#project-structure)
2. [CACAD Setup](#cacad-setup)
  i.    [Installation](#installation)
  ii.   [Dataset](#dataset)
  iii.  [Preprocessing](#preprocessing)
3. [Counseling]
  i.    [NQCP](#)
  ii.   [Offensive Question Detection] 
  iii.  [Abuse Detection]
4. [Citation](#citation)
---

## Project Structure
---
```plaintext
├── app/                         
│   ├── CACAD_32B.py
│   ├── CACAD_Gradio.py
│   └── NQCP.py
├── configs/
│   └── base_config.yaml             
├── data/
│   ├── raw/ # [User-provided] Raw data provided by the user (.json format) 
│   │   └── …         
│   └── processed/ # [Will be generated] processed files.
│       ├── labeled_dataset/     (train/*, test/*, val/*)
│       ├── finetuning_dataset/   (train.json, test.json, val.json)
│       └── offensive_dataset/    (train.csv, test.csv, val.csv)
├── prompts/
│   ├── cluster_details/ # [Will be generated]   
│   └── counseling/     
│       └── …            
├── shells/  
│   └── …
├── src/
│   ├── abuse_detection/
│   │   ├── MLC/                  
│   │   └── uncertainty/
│   ├── counseling/
│   │   ├── NQCP/                 
│   │   └── offensive_question/
│   └── processing/
│       ├── Clustering/ 
│       │   ├── utils/
│       │   └── main.py
│       ├── gen_ft_dataset.py
│       └── preprocess_raw.py
├── .gitignore
└── README.md
```

---
## CACAD Setup
---
### Installation

```bash
cd Conversational-AI-for-Child-Abuse-Detection
```

```bash
pip install torch 
pip install -r requirements.txt
```

Set up your output directory and cache directory in `config/base_config.yaml`

### Dataset

We conducted CACAD using the following datsets:
[Child and adolecent counseling data](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71680)

> Once downloaded, unzip both the TL_out and VL_out zip files into the same `data/raw` folder.
> The resulting structure should look like this:

```plaintext
├── data/
│   ├── raw/ 
│   │   ├──0001.json
│   │   ├──0002.json
│   │   └── …         
```

You can also use your own dataset, as long as it follows the same data structure below:

Data structure details

```bash
{
    "list": [
        {
            "항목" : "category1",
            "label" : 0 ,
            "audio": [
                {
                    "type": "Q"
                    "text": "sample text"
                },
                {
                    "type": "A"
                    "text": "sample text"
                },...
            ]
        },...
        {
            "항목" : "category4",
            "label" : 1 ,
            "audio": [
                {
                    "type": "Q"
                    "text": "sample text"
                },
                {
                    "type": "A"
                    "text": "sample text"
                },...
            ]
        }
    ],
    "ground_truth": [0,1,0,1]
}
```

### Preprocessing
##### Step 1:

```bash
sh shells/preprocess/preprocess_raw.sh
```

##### Step 2:

```bash
sh shells/preprocess/gen_ft_dataset.sh
```


##### Step 3:

```bash
sh shells/preprocess/add_clustering_result.sh
```



## Citation

---

```bibtex
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
```

