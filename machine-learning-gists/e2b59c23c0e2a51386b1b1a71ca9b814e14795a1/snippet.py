#! /usr/bin/env python

# -*- coding: utf-8 -*-

import os
import sys

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

captchas_path = "./captchas/"
captcha_key = "some_value"

if not os.path.isdir(captchas_path):
    os.makedirs(captchas_path)
     
# open TXT files with \n separated values to be sent to google
with open(sys.argv[1], "r") as input_file:
    keys = [str(x).strip() for x in input_file]
    
    
def tesseract(captcha_image):

    try:
        print("Running tesseract operation on --> {}".format(captcha_image))

        subprocess.call(['tesseract', captcha_image, 'output', '-psm', '8'])

        file_name = os.path.join(captcha_image.split('.')[0])

        with open(file_name + '.txt', 'w') as file:
            global captcha_value

            value_lines = file.read().replace(" ", "").split('\n')
            captcha_value = value_lines[0]

            print('Captcha for this image file is --> {}'.format(captcha_value))

    except Exception as e:
        print(e)
        
        
def captcha_black():
    listdir = os.listdir("./captchas/")

    for photo in listdir:
        if photo.endswith(".png"):
            img = Image.open("./captchas/" + photo)
            img = img.convert("RGBA")

            pixdata = img.load()

            for y in range(img.size[1]):
                for x in range(img.size[0]):
                    if pixdata[x, y][0] < 90:
                        pixdata[x, y] = (0, 0, 0, 255)

            for y in range(img.size[1]):
                for x in range(img.size[0]):
                    if pixdata[x, y][1] < 136:
                        pixdata[x, y] = (0, 0, 0, 255)

            for y in range(img.size[1]):
                for x in range(img.size[0]):
                    if pixdata[x, y][2] > 0:
                        pixdata[x, y] = (255, 255, 255, 255)

            black_captcha = "./captchas/{}-black.gif".format(photo)
            img.save(black_captcha, "GIF")

            im_orig = Image.open(black_captcha)
            big = im_orig.resize((1000, 500), Image.BILINEAR)

            ext = ".tif"
            big.save("./captchas/{}-black-tiff".format(photo) + ext)


def get_captcha(driver, img, captcha_number):
    img_captcha_base64 = driver.execute_async_script(
        """
        var ele = arguments[0], callback = arguments[1];
        ele.addEventListener('load', function fn(){
        ele.removeEventListener('load', fn, false);
        var cnv = document.createElement('canvas');
        cnv.width = this.width; cnv.height = this.height;
        cnv.getContext('2d').drawImage(this, 0, 0);
        callback(cnv.toDataURL('image/png').substring(22));
        }, false); ele.dispatchEvent(new Event('load'));
        """, img
    )

    with open(r"./captchas/captcha-{}.png".format(captcha_number), 'wb') as f:
        f.write(base64.b64decode(img_captcha_base64))

        
def main():
    driver = webdriver.Chrome()

    captcha_number = 0

    driver.get("http://www.google.com/")
    driver.maximize_window()
    driver.implicitly_wait(10)
    driver.set_page_load_timeout(10)

    for keyword in keys:
        captcha_number += 1

        WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.NAME, "q"))).send_keys(keyword, Keys.ENTER)

        try:
            # if captcha
            img = WebDriverWait(driver, 1.5).until(EC.presence_of_element_located((By.XPATH, "/html/body/div/img")))

            if img:
                while True:
                    get_captcha(driver, img, captcha_number)

                    WebDriverWait(driver, 1.5).until(EC.presence_of_element_located((By.NAME, "captcha"))).send_keys(captcha_key)
                    WebDriverWait(driver, 1.5).until(EC.presence_of_element_located((By.NAME, "submit"))).click()
            else:
                continue

        except:
            WebDriverWait(driver, 2).until(EC.presence_of_element_located((By.NAME, "q"))).clear()

    driver.quit()


main()
