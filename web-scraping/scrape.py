from bs4 import BeautifulSoup
import json
import requests
import time


headers = {
    "User-Agent": "Graduate-Student-Trynna-Get-Data-For-My-Project-Thanks-2"
}

with open("so-question-links.json", "r") as f:
    links = json.load(f)
    links = links[6000:]

all_posts = []
l_idx = 1
for question in links:
    try:

        page = requests.get(question['url'], {'timeout': 10}, headers=headers)
        print("scraping {} - status code {} - ({}/{})".format(question['url'], page.status_code, l_idx, len(links)))
        soup = BeautifulSoup(page.content, "html.parser")

        post_body = soup.find("div", class_="s-prose js-post-body")
        post_body = post_body.get_text(separator=" ", strip=True)

        data = {}
        data['question'] = soup.find(id="question-header").find("a", class_="question-hyperlink").get_text(strip=True)
        data['body'] = post_body
        data['upvote'] = soup.find("div", class_="js-vote-count").get_text(strip=True)
        data['url'] = question['url']
        answers = []
        answer_html = soup.find(id="answers")
        upvote_html = answer_html.find_all("div", class_="js-vote-count")
        answer_html = answer_html.find_all("div", class_="s-prose js-post-body")

        idx = 0
        for ans in answer_html:
            answer_body = ans.get_text(separator=" ", strip=True)
            answer_upvote = upvote_html[idx].get_text(strip=True)
            answers.append({"body": answer_body, "upvote": answer_upvote})
            idx += 1

        data['answers'] = answers
        all_posts.append(data)
        time.sleep(2)
        l_idx += 1
    except Exception as e:
        l_idx += 1
        print(f"exception in url {question['url']}")

with open(f"so-posts-{time.time_ns()}.json", "w") as f:
    json.dump(all_posts, f, indent=2)
