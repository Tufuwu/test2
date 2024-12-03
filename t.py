from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
# 设置 WebDriver 的路径
edge_driver_path = r'D:\heel_foo\msedgedriver.exe'
service = Service(executable_path=edge_driver_path)

# 配置 WebDriver
driver = webdriver.Edge(service=service)

try:
    # 打开淘宝登录页面
    driver.get('https://login.taobao.com/')

    # 等待用户手动登录
    time.sleep(20)

    # 打开购物车页面
    driver.get('https://cart.taobao.com/cart.htm')

    # 等待购物车页面加载
    WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.CSS_SELECTOR, '.cart-item')))

    # 找到购物车中的商品
    items = driver.find_elements(By.CSS_SELECTOR, '.cart-item')

    # 假设选择第一个商品
    if items:
        item = items[0]
        item.find_element(By.CSS_SELECTOR, '.J_Choose').click()

        # 点击结算按钮
        checkout_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, '.J_GoToCheckout'))
        )
        checkout_button.click()

        # 等待结算页面加载

        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, '.J_SubmitOrder'))
        )

        # 提交订单
        submit_button = driver.find_element(By.CSS_SELECTOR, '.J_SubmitOrder')
        submit_button.click()

        print("抢购成功！")
    else:
        print("购物车中没有商品！")

except Exception as e:
    print(f"发生错误: {e}")

finally:
    # 关闭浏览器
    driver.quit()