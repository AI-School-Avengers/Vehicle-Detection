import os
from selenium.webdriver.support import expected_conditions as EC

from selenium import webdriver
import urllib.request
import time

from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait

driver = webdriver.Chrome('D:/chromedriver')
url = 'http://mdcar.kr/search/list.mdc?ct=-2&b=%EC%A0%9C%EC%A1%B0%EC%82%AC&mg=%EB%AA%A8%EB%8D%B8&m=%EC%83%81%EC%84%B8%2B%EB%AA%A8%EB%8D%B8&cg=%EB%93%B1%EA%B8%89&c=&tt=&ft=&yb=&ye=&area=&cp=&cp2=&ac=&ip=&id=&fx=&ck=&lp=&pb=&pe=&mb=&me=&o1=&o2=&o3=&o4=&o5=&or=4&pg={}'

# wait until someid is clickable
wait = WebDriverWait(driver, 10)

pic_num = 0

for pg in range(9,16): # 페이지 순서대로 들어가기
    driver.get(url.format(pg+1))
    print("page" , pg+1)
    driver.implicitly_wait(30)   #로딩 대기
    # wait.until(EC.element_to_be_clickable((By.CLASS_NAME, 'price')))
    time.sleep(1.5)

    for pic in range(2, 17) : # 자동차 목록 순서대로 들어가기
        print("pic" , pic)
        # page_list_list > li:nth-child(2) > div > div > a
        # page_list_list > li:nth-child(3) > div > div > a
        driver.find_element_by_css_selector('#page_list_list > li:nth-child({}) > div > div > a > div.price'.format(pic)).click()
        time.sleep(2.5)
        # wait.until(EC.element_to_be_clickable((By.ID, 'img')))    # img 뜨기전에 안넘어가도록
        for i in range(3):    # 슬라이드에서 1~4번째 사진 빼오기
            images = driver.find_elements_by_css_selector("#img")

            for img in images:
                driver.find_element_by_css_selector('#photoNxt').click()
                imgURL = img.get_attribute('src')
                print(imgURL)

            pic_num += 1

            if not os.path.exists('D:/moldeonCar'):
                os.makedirs('D:/moldeonCar')

            urllib.request.urlretrieve(imgURL, 'D:/moldeonCar/'+ str(318+pic_num) + ".jpg")
        driver.back()
        time.sleep(1.5)
        # wait.until(EC.element_to_be_clickable((By.CLASS_NAME, 'price')))

driver.close()
