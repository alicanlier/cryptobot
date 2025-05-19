from seleniumwire import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
import decimal
import time
from src.binance.futures_prices import round_to_4digits, round_long_decimals


def call_driver(url):
    # Set the path to the ChromeDriver executable
    chrome_driver_path = 'C:/Users/A/Desktop/my-PT/chromedriver-win32/chromedriver.exe'
    # chrome_options = Options()
    # service = Service(executable_path=chromeDriver)
    # driver = webdriver.Chrome(service=service, options=chrome_options)
    driver = webdriver.Chrome(chrome_driver_path)
    driver.get(url)

    existing_session_id = driver.session_id
    existing_session_cookies = driver.get_cookies()

    driver.quit()

    options = webdriver.ChromeOptions()
    options.debugger_address = 'localhost:9222'
    driver = webdriver.Chrome(options)

    # Attach to the existing session
    driver.session_id = existing_session_id
    for cookie in existing_session_cookies:
        driver.add_cookie(cookie)

    #####################

    # chrome_options = webdriver.ChromeOptions()
    # chrome_options.add_argument("--remote-debugging-port=9222")  # Enable remote debugging port
    # service = Service(chromeDriver)
    # service.start()
    #
    # # Get the existing session ID and session cookies
    # existing_session_id = None
    # existing_session_cookies = None
    #
    # for request in webdriver.Chrome.get_all_sessions(service.service_url):
    #     if "web.whatsapp.com" in request["url"]:
    #         existing_session_id = request["sessionId"]
    #         existing_session_cookies = request["cookies"]
    #         break
    #
    # # Quit the Selenium service
    # service.stop()
    #
    # # Set up Chrome WebDriver with the existing session
    # chrome_options.add_experimental_option("debuggerAddress", "127.0.0.1:9222")
    # driver = webdriver.Chrome(executable_path=chromeDriver, options=chrome_options)
    # driver.session_id = existing_session_id
    #
    # for cookie in existing_session_cookies:
    #     driver.add_cookie(cookie)

    #################

    return driver


def crawl_url(url):
    # Load the webpage
    driver = call_driver(url)
    driver.get(url)

    time.sleep(10)  # Allow time for manual login

    # Find the input field for typing messages
    css_selector = '#main > footer > div._2lSWV._3cjY2.copyable-area > div > span:nth-child(2) > div > div._1VZX7 > div._3Uu1_ > div.g0rxnol2.ln8gz9je.lexical-rich-text-input > div.to2l77zo.gfz4du6o.ag5g9lrv.bze30y65.kao4egtt > p'
    input_field = driver.find_element(By.CSS_SELECTOR, css_selector)
    print(input_field.text)

    # Type your message
    # input_field.send_keys('Hello from Selenium!' + Keys.ENTER)

    print("Message sent successfully!")

    # Close the WebDriver
    driver.quit()




if __name__ == '__main__':
    url = 'https://web.whatsapp.com/'
    # crawl_url(url)
    num = 10001.1
    # print(round_to_4digits(num))
    round_long_decimals(num)
