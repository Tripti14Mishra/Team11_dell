# from nose.tools import assert_equal, assert_true
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from behave import given,when,then
import time

def highlight(element):
    """Highlights (blinks) a Selenium Webdriver element"""
    driver = element._parent
    def apply_style(s):
        driver.execute_script("arguments[0].setAttribute('style', arguments[1]);",
                              element, s)
    original_style = element.get_attribute('style')
    apply_style("border: 3px solid red;")
    time.sleep(0.3)
    apply_style(original_style)

# Given that I'm logged into the application as a Manager
@given("that I land on dell.com")
def user_enters_webpage(context):
    context.browser.get("http://127.0.0.1:5000/")


# When I land on the Settings & Configuration window
@when("I look at page")
def look(context):
    time.sleep(1)
    
# And I select SSD - ENTERPRISE as commodity
@then("i should see recommendations")
def seerec(context):
    xpathrec = str("//*[@id='appenddiv']/div[1]/h5")
    rec=context.browser.find_element(By.XPATH, xpathrec)
    
    highlight(rec)
    time.sleep(1.5)
    rec=rec.text
    assert(len(rec)>0)


@when("I enter mini and maxi")
def enter(context):
    xpathmin = str("//*[@id='min']")
    xpathmax = str("//*[@id='max']")
    
    mini = context.browser.find_element(By.XPATH, xpathmin)
    maxi = context.browser.find_element(By.XPATH, xpathmax)
    highlight(mini)
    mini.send_keys("100")

    highlight(maxi)
    maxi.send_keys("2000")

# And I click on Apply
@when("I click ok")
def clickok(context):
    button = context.browser.find_element(By.XPATH,"//*[@id='sortprice']")
    highlight(button)
    time.sleep(1)
    # highlight(button)
    button.click()
    time.sleep(1.8)

