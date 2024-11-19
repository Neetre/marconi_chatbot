from bs4 import BeautifulSoup
import requests

website="https://cercalatuascuola.istruzione.it/cercalatuascuola/istituti/VRTF03000V/guglielmo-marconi/ptof/naviga/"
div_class="panel-body"


def get_html(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raises an HTTPError if the response status code is 4XX or 5XX
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL {url}: {e}")
        return None


def scrap_html(html):
    if html is None:
        print("No HTML to parse")
        return None
    try:
        soup = BeautifulSoup(html, 'html.parser')
        texts = soup.find_all('div', class_='panel-body')
        return texts

    except Exception as e:
        print(f"Error parsing HTML: {e}")
        return None

def main():
    html_txt = get_html(website)
    print(html_txt)
    text = scrap_html(html_txt)
    print(text)
    
if __name__ == "__main__":
    main()
