import requests
import time
import re
from bs4 import BeautifulSoup
import xml.etree.cElementTree as ET


def spider(api):
    page =101
    headers = requests.utils.default_headers()
    headers.update({
        'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0',
    })
    li = []
    while(page <= 130 ):
        url = "https://stackoverflow.com/search?page="+str(page)+"&tab=Relevance&q="+api
        sourcecode = requests.get(url,headers=headers)
        text = sourcecode.text
        stacksoup = BeautifulSoup(text,"html.parser")
        for divs in stacksoup.find_all("div",{"class":"result-link"}):
            for links in divs.find_all("a"):
                link = "https://www.stackoverflow.com"+links.get("href")
                li.append(link)
        time.sleep(2)
        page += 1

    return li

def crawlQA(links,root):
    headers = requests.utils.default_headers()
    headers.update({
        'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0',
    })
    data = []

    for link in links:
        doc = ET.SubElement(root,"doc")
        data = ET.SubElement(doc,"data")
        sentiment = ET.SubElement(doc,"sentiment")
        string = ""
        sourcecode = requests.get(link,headers=headers)
        text = sourcecode.text
        answersoup = BeautifulSoup(text,"html.parser")
        questionDiv = answersoup.find("div", {"class": "question"})
        answer = answersoup.find("div",{"class":"answercell post-layout--right"})



        if(questionDiv != None):
            QSubDiv = questionDiv.find("div",{"class":"post-text"}).findChildren()
            for Q in QSubDiv:
                if (Q.name == "pre"):
                    continue
                else:
                    string += parseData(Q.text)

        if (answer != None):
            ans = answer.find("div",{"class":"post-text"}).findChildren()
            for A in ans:

                if(A.has_attr("class") and A["class"][0]=="lang-py prettyprint prettyprinted"):
                    continue
                else:
                    string += parseData(A.text)

        data.text = string


    return root

# deprected
def getListofSocialAPI(website):

    socialAPI = []
    headers = requests.utils.default_headers()
    headers.update({
        'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0',
    })
    sourcecode = requests.get(website, headers=headers)
    text = sourcecode.text
    stacksoup = BeautifulSoup(text, "html.parser")
    div = stacksoup.find("div", {"class": "post_text_inner"})

    for links  in div.find_all("a", {"rel" : "noopener noreferrer"}):
        if((links.text)[0].isdigit()):
            link = "".join(re.findall("[a-zA-Z]+", links.text))
            socialAPI.append(link)
            print(link)


    return socialAPI

def parseData(data):

    lines = data.split("\n")
    reviews =""
    for line in lines:

        expression = r"^https?:\/\/.*[\r\n]*"
        linksRemoved = re.sub(expression, "", line, flags=re.MULTILINE)
        longWords = "\b\p{L}{20,}"
        newString = re.sub(longWords, "", linksRemoved, flags=re.MULTILINE)
        if (len(newString) > 40):
            pattern = r"[^\x00-\x7F]+"
            patternAN = r"[^a-zA-Z \.]"
            temp = re.sub(pattern," ", linksRemoved)
            temp2 = re.sub(patternAN,"",temp)
            reviews += temp2.strip()
    return reviews

def initiateSpider():


    webapi = ["Facebook","Tumblr","WhatsApp","Instagram","Twitter","Skype","Viber","Snapchat","Pinterest",
              "LinkedIn","Telegram","Reddit","Foursquare","Myspace","YouTube","Vine","Flickr","Rediff",
              "Twitter%20Streaming","Meetup","WeChat","GooglePlus","New%20York%20Times","Stackoverflow","Yahoo","QQ"]
    root = ET.Element("root")
    for api in webapi:
        time.sleep(1000)
        links = spider(api+"%20API")
        print(api)
        root = crawlQA(links,root)


    tree = ET.ElementTree(root)
    tree.write("reviews2.xml")


initiateSpider()