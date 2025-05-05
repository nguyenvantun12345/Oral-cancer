
from bs4 import BeautifulSoup
import pandas as pd
import requests

# Crawl dữ liệu từ website
url = "https://oralcancerfoundation.org/dental/oral-cancer-images/"
response = requests.get(url)
data = []

if response.status_code == 200:
    soup = BeautifulSoup(response.text, "html.parser")
    crawl = soup.find_all("div", class_="l290-12 image-box")
    
    for idx, i in enumerate(crawl, start=1):
        dic = {
            "Index": idx,
            "image": "https://oralcancerfoundation.org" + i.select_one("img").get("src"),
            "Diagnois": i.select_one("h5").text.strip(),
            "KetLuan": "",
            "Group": ""
        }
        data.append(dic)
        
else:
    print("Failed to retrieve the webpage")

df = pd.DataFrame(data)
df.set_index("Index", inplace=True)  # Đặt cột "Index" làm chỉ mục
df.to_excel("scraping.xlsx")
