from selenium import webdriver
from selenium.webdriver.chrome.options import Options

import os
os.environ["PATH"] += os.pathsep + '../features'

def before_all(context):
    # options = Options()
    # options.headless = True
    context.browser = webdriver.Chrome('C:/chromedriver.exe')

    context.browser.set_page_load_timeout(200)
    context.browser.maximize_window()

def after_all(context):
    context.browser.quit()