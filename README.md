# Movie review generator

## Description

**Naver Movie Review Data Based Review Generation Tool Using openAI API**  
openAI api를 활용한 네이버 영화 리뷰 데이터 기반 평론 생성 도구

## Dataset Description

Movie review datasets collected from Naver Movie, which is the largest Korean movie review website. These datasets were obtained through web scraping and have been made available at [Mendeley Data](https://data.mendeley.com/datasets/jb5knzh8yv/6).

## How to run

1. Get API key from Naver Cloud Platform  
   1\) Sign up for [Naver Cloud Platform](https://console.ncloud.com/naver-service/application) and create `Clova Summary Application`  
   2\) Copy your API key
    - client ID
    - client secret
2. Set environment variable  
   1\) Create `.env` file in the root directory and add below line
    ```bash
    client_id = {YOUR_CLIENT_ID}
    client_secret = {YOUR_CLIENT_SECRET}
    ```
