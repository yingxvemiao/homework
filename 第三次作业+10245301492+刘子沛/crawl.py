# -*- coding: utf-8 -*-
"""
ESI Institutions × Research Fields 批量下载（XLS）
依赖：pip install selenium
说明：复用本机 Chrome 登录态（默认“Default”），逐学科勾选→下载XLS→以学科名命名文件→清除过滤继续。
"""

import os, time, glob, shutil, datetime, platform
from typing import Optional, List
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC

ESI_URL = "https://esi.clarivate.com/IndicatorsAction.action?app=esi&Init=Yes"
DL_DIR  = os.path.abspath("esi_downloads")
TIMEOUT = 60

def detect_chrome_user_data_dir() -> Optional[str]:
    sysname = platform.system().lower()
    if "windows" in sysname:
        p = os.path.join(os.environ.get("LOCALAPPDATA",""), "Google", "Chrome", "User Data")
        return p if os.path.isdir(p) else None
    if "darwin" in sysname:
        p = os.path.expanduser("~/Library/Application Support/Google/Chrome")
        return p if os.path.isdir(p) else None
    for p in ["~/.config/google-chrome", "~/.config/chromium"]:
        q = os.path.expanduser(p)
        if os.path.isdir(q): return q
    return None

def init_driver() -> webdriver.Chrome:
    os.makedirs(DL_DIR, exist_ok=True)
    opt = webdriver.ChromeOptions()
    ud = detect_chrome_user_data_dir()
    if ud:
        opt.add_argument(f"--user-data-dir={ud}")
        opt.add_argument("--profile-directory=Default")
    opt.add_experimental_option("prefs", {
        "download.default_directory": DL_DIR,
        "download.prompt_for_download": False,
        "safebrowsing.enabled": True
    })
    d = webdriver.Chrome(options=opt)
    d.set_page_load_timeout(90)
    return d

def w(drv, by, sel, to=TIMEOUT):
    return WebDriverWait(drv, to).until(EC.presence_of_element_located((by, sel)))

def wc(drv, by, sel, to=TIMEOUT):
    return WebDriverWait(drv, to).until(EC.element_to_be_clickable((by, sel)))

def unique_path(base_no_ext: str, ext: str) -> str:
    p = os.path.join(DL_DIR, base_no_ext + ext)
    if not os.path.exists(p): return p
    i = 1
    while True:
        q = os.path.join(DL_DIR, f"{base_no_ext}_{i}{ext}")
        if not os.path.exists(q): return q
        i += 1

def sanitize(name: str) -> str:
    s = "".join(c for c in name if c.isalnum() or c in " _-").strip().replace(" ", "_")
    return s or "ESI_Field"

def wait_download(before: List[str], to=240) -> Optional[str]:
    end = time.time() + to
    while time.time() < end:
        cur = glob.glob(os.path.join(DL_DIR, "*"))
        new = [f for f in cur if f not in before and not f.endswith(".crdownload")]
        if new:
            f = max(new, key=os.path.getctime)
            if not os.path.exists(f + ".crdownload"):
                return f
        time.sleep(0.4)
    return None

def set_results_institutions(drv):
    sel = w(drv, By.XPATH, "//label[contains(.,'Results List')]/following::select[1]")
    Select(sel).select_by_visible_text("Institutions")
    WebDriverWait(drv, TIMEOUT).until(
        EC.presence_of_element_located((By.XPATH, "//*[contains(@class,'results') or contains(@id,'results')]"))
    )

def open_rf_panel(drv):
    wc(drv, By.XPATH, "//button[contains(.,'Add Filter')]").click()
    wc(drv, By.XPATH, "//a[contains(.,'Research Fields')]").click()
    w(drv, By.XPATH, "//div[contains(@class,'modal') or contains(@class,'dialog')]//label[.//input[@type='checkbox']]")

def list_fields(drv) -> List[str]:
    items = drv.find_elements(By.XPATH,
        "//div[contains(@class,'modal') or contains(@class,'dialog')]//label[.//input[@type='checkbox']]"
    )
    names = []
    for it in items:
        t = it.text.strip()
        if t: names.append(t)
    return names

def apply_field(drv, name: str):
    wc(drv, By.XPATH,
       f"//div[contains(@class,'modal') or contains(@class,'dialog')]//label[normalize-space()='{name}']//input[@type='checkbox']"
    ).click()
    wc(drv, By.XPATH, "//button[contains(.,'Apply')]").click()
    WebDriverWait(drv, TIMEOUT).until(
        EC.presence_of_element_located((By.XPATH, "//*[contains(@class,'results') or contains(@id,'results')]"))
    )

def clear_field(drv, name: str):
    # 优先点结果页过滤标签的“×”
    try:
        wc(drv, By.XPATH,
           f"//div[contains(@class,'filter') or contains(@class,'chips') or contains(@id,'filters')]"
           f"//*[contains(normalize-space(),'{name}')]/following::*[self::button or self::a][1]"
        ).click()
        WebDriverWait(drv, TIMEOUT).until(
            EC.presence_of_element_located((By.XPATH, "//*[contains(@class,'results') or contains(@id,'results')]"))
        )
        return
    except:
        pass
    # 回到面板取消勾选
    wc(drv, By.XPATH, "//button[contains(.,'Add Filter')]").click()
    wc(drv, By.XPATH, "//a[contains(.,'Research Fields')]").click()
    wc(drv, By.XPATH,
       f"//div[contains(@class,'modal') or contains(@class,'dialog')]//label[normalize-space()='{name}']//input[@type='checkbox']"
    ).click()
    wc(drv, By.XPATH, "//button[contains(.,'Apply')]").click()
    WebDriverWait(drv, TIMEOUT).until(
        EC.presence_of_element_located((By.XPATH, "//*[contains(@class,'results') or contains(@id,'results')]"))
    )

def download_xls(drv, field_name: str) -> Optional[str]:
    before = glob.glob(os.path.join(DL_DIR, "*"))
    wc(drv, By.XPATH, "//button[contains(.,'Download') or @title='Download']").click()
    wc(drv, By.XPATH, "//a[contains(.,'XLS') or contains(.,'xls')]").click()
    f = wait_download(before)
    if not f: return None
    base = sanitize(field_name)                 # 文件名以学科名为核心
    target = unique_path(base, ".xls")          # 若重名自动加后缀
    shutil.move(f, target)
    return target

def main():
    drv = init_driver()
    try:
        drv.get(ESI_URL)
        time.sleep(2)
        set_results_institutions(drv)
        open_rf_panel(drv)
        fields = list_fields(drv)
        # 关闭面板（若存在“Close/Done”）
        try:
            wc(drv, By.XPATH, "//button[contains(.,'Close') or contains(.,'Done')]").click()
            WebDriverWait(drv, 10).until(EC.invisibility_of_element_located((By.XPATH, "//div[contains(@class,'modal')]")))
        except:
            pass

        print(f"检测到学科 {len(fields)} 个")
        for i, name in enumerate(fields, 1):
            print(f"[{i}/{len(fields)}] {name}")
            wc(drv, By.XPATH, "//button[contains(.,'Add Filter')]").click()
            wc(drv, By.XPATH, "//a[contains(.,'Research Fields')]").click()
            apply_field(drv, name)
            saved = download_xls(drv, name)
            print("保存：", saved if saved else "下载失败")
            clear_field(drv, name)
        print("完成，目录：", DL_DIR)
    finally:
        drv.quit()

if __name__ == "__main__":
    main()
