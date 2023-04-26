import requests
from bs4 import BeautifulSoup

url='https://en.wikipedia.org/wiki/ChatGPT'

# save text from the url to a txt file
with open(url.split('/')[-1]+".txt", "w", encoding="UTF-8") as f:
    # get the text from the URL using BeautifulSoup
    soup = BeautifulSoup(requests.get(url).text, "html.parser")

    # save paragraphs
    for i in soup.select('p'):
        f.write(i.get_text())
