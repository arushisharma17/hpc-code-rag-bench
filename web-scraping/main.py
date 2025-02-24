import re
import requests
from bs4 import BeautifulSoup
import json
import time


base_url = "https://stackoverflow.com"
max_page = 134

url = "https://stackoverflow.com/questions/tagged/openmp?sort=MostVotes&edited=true&pagesize=50"
headers = {
    "User-Agent": "Graduate-Student-Trynna-Get-Data-For-My-Project-Thanks"
}

all_question_links = []

for i in range(max_page):
    pageNumber = i + 1
    search_url = url + '&page={}'.format(pageNumber)
    print('scraping {}...'.format(search_url))
    page = requests.get(search_url, {'timeout': 10}, headers=headers)
    soup = BeautifulSoup(page.content, "html.parser")
    question_links = soup.find_all(href=re.compile("^/questions/[0-9]+/"))

    for q in question_links:
        all_question_links.append({"url": base_url + q.attrs['href']})
    time.sleep(1.5)

with open("so-question-links.json", "w") as f:
    json.dump(all_question_links, f, indent=4)
