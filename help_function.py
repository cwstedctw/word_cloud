import csv
import re
import pandas as pd
from ckiptagger import WS, POS, NER, construct_dictionary, data_utils  # 中文斷詞、詞性標註、命名實體識別
from config import *

def download_ckip_data():
    """
    檢查 ./data 目錄是否存在且非空，若不存在或為空則下載 CKIP 資料。
    此函數會下載約 2GB 的資料至 ./data.zip，並解壓縮至 ./data/ 目錄。
    """
    import os
    if not os.path.exists("./data") or len(os.listdir("./data")) == 0:
        # 從 Google Drive 下載 CKIP 資料
        # 檔案大小約 2GB，將下載至 ./data.zip 並解壓縮至 ./data/
        # data_utils.download_data_url("./")  # 從 IIS 伺服器下載
        data_utils.download_data_gdown("./")  # 從 Google Drive 下載

#-------------------------------------------------------------------------------

def load_removes(file_path: str) -> list:
    """
    從指定檔案載入需要被移除的字詞清單。

    參數:
    file_path (str): 包含需要移除字詞的檔案路徑。

    回傳:
    list: 需要被移除的字詞清單。
    """
    removes = []  # 初始化儲存字詞的列表
    with open(file_path, encoding='utf-8-sig', newline='\r\n') as csvfile:
        spamreader = csv.reader(csvfile)
        for row in spamreader:
            # 使用列表生成式移除列表中的空字串
            result = [word for word in row if word != ""]
            removes += result
    return removes

#-------------------------------------------------------------------------------

def load_synonyms(synonym_file: str) -> list:
    """
    從指定的 CSV 文件載入同義詞詞典，並處理資料格式。
    
    參數:
    synonym_file (str): 同義詞 CSV 文件的路徑。
    
    回傳:
    list: 處理後的同義詞列表，每個子列表包含一組同義詞。
    """
    # 讀取同義詞 CSV 文件，將其轉置並重置索引
    synonym_df = pd.read_csv(synonym_file).T.reset_index()
    
    # 將每行轉換為列表並移除 NaN 值
    # 每個列表代表一組同義詞
    synonymes = [row.dropna().tolist() for _, row in synonym_df.iterrows()]

    # 處理同義詞中的空格
    result = []
    for string_list in synonymes:
        new_list = []
        for string in string_list:
            # 移除每個詞中的空格
            new_list.append(string.replace(' ', ''))
        result.append(new_list)
    
    # 更新同義詞列表
    synonymes = result
    #print(f"\n同義詞詞典 \n{synonymes}")
    
    return synonymes


def read_coldata(df: pd.DataFrame, col_no: int) -> pd.DataFrame:
    """
    從 pandas DataFrame 中讀取指定欄位編號前綴的資料。

    參數:
    col_no (int): 欲搜尋的欄位編號前綴。

    回傳:
    pandas.DataFrame: 包含符合指定欄位編號前綴的資料。

    範例:
    如果 DataFrame 'df' 的欄位為 '1. Name', '2. Age', '3. Address'，
    呼叫 read_coldata(2) 將回傳只包含 '2. Age' 欄位的資料。

    注意:
    - 此函數假設 DataFrame 'df' 在函數呼叫的範圍內可存取。
    - 欄位編號前綴會被轉換為字串，並附加一個句點以匹配欄位名稱中的確切模式。
    """

    # 將欄位編號轉換為字串並附加句點以建立搜尋模式
    search_pattern = str(col_no) + "."

    # 找出 DataFrame 中符合搜尋模式的欄位
    col = [column for column in df.columns if column.startswith(search_pattern)]

    # 選取符合的欄位資料並移除空值
    col_data = df[col[0]].dropna()

    # 回傳結果作為列表
    return col_data.to_list()

def take_wordCounts(data: pd.DataFrame):
    """
    對資料進行中文斷詞並統計詞頻。

    參數:
    data (pd.DataFrame): 欲處理的資料。

    回傳:
    dict: 詞頻統計結果。
    """
    text = []
    for row in data:
        if type(row) is str:
            res, n = re.subn('\s+', '', row)  # 移除空白字元
            text.append(res)

    # 中文斷詞
    ws = WS(data_folder_path, disable_cuda=False)
    pos = POS(data_folder_path, disable_cuda=False)
    ner = NER(data_folder_path, disable_cuda=False)

    # 建立句子列表
    sentence_list = text

    # 載入同義詞詞典
    synonymes = load_synonyms(synonym_file)
    word_to_weight = {string: 2 for string_list in synonymes for string in string_list if len(string) > 2}

    # 使用 construct_dictionary 函數建立詞典
    dictionary = construct_dictionary(word_to_weight)

    # 進行斷詞
    word_sentence_list = ws(
        sentence_list,
        coerce_dictionary=dictionary,  # 強制使用詞典中的詞
    )
    pos_sentence_list = pos(word_sentence_list)
    entity_sentence_list = ner(word_sentence_list, pos_sentence_list)

    # 統計斷詞結果
    words = []
    for word in word_sentence_list:
        words = words + word
    from collections import Counter
    word_counts = dict(Counter(words).most_common())
    return word_counts


def make_removedFile(counts, col_no: int):
    """
    根據排除詞檔案移除指定詞並輸出結果。

    參數:
    counts (dict): 詞頻統計結果。
    col_no (int): 欲處理的欄位編號。
    """
    removeds = {}

    # 從排除詞檔案中載入詞彙
    removes = load_removes(remove_file)
    
    # 處理每個需要移除的詞
    for rmw in removes:
        try:
            no = counts.pop(rmw)
            removeds[rmw] = no
        except:
            ...

    # 輸出排除詞檔案
    removeout_path = word_cloud_path + f'/排除詞_{col_no}.txt'
    with open(removeout_path, 'w', encoding='utf-8') as f:  # 指定 UTF-8 編碼
        for key, value in removeds.items():
            f.write(f'{key}:{value}\n')

    # 輸出排除詞文字雲檔案
    removewdcnt_path = word_cloud_path + f'/排除詞文字雲_{col_no}.txt'
    with open(removewdcnt_path, 'w', encoding='utf-8') as f:  # 指定 UTF-8 編碼
        for key, value in removeds.items():
            for i in range(value):
                f.write(f'{key} ')


def make_synonymFile(counts, col_no: int):
    """
    根據同義詞檔案合併同義詞並輸出結果。

    參數:
    counts (dict): 詞頻統計結果。
    col_no (int): 欲處理的欄位編號。
    """
    synonymws = {}

    # 從同義詞檔案中載入同義詞
    synonyms = load_synonyms(synonym_file)
    
    for sw in synonyms:
        totalw = 0
        if not sw:  # 跳過空的同義詞組
            continue
            
        kwywd = sw[0]  # 第一個詞作為關鍵詞
        
        for w in sw:
            try:
                if w in counts:
                    totalw += counts[w]
                    counts.pop(w)  # 移除詞以避免重複計算
            except:
                pass
                
        if totalw != 0:
            synonymws[kwywd] = totalw

    # 輸出同義詞檔案
    synonymout_path = word_cloud_path + f'/同義詞_{col_no}.txt'
    with open(synonymout_path, 'w', encoding='utf-8') as f:  # 指定 UTF-8 編碼
        for key, value in synonymws.items():
            f.write(f'{key}:{value}\n')

    # 輸出同義詞文字雲檔案
    synonymws_path = word_cloud_path + f'/同義詞文字雲_{col_no}.txt'
    with open(synonymws_path, 'w', encoding='utf-8') as f:  # 指定 UTF-8 編碼
        for key, value in synonymws.items():
            for i in range(value):
                f.write(f'{key} ')


# 主程式入口
if __name__ == '__main__':
    
    # 讀取所需的欄位資料
    df = pd.read_excel(data_file)
    col  = 23
    word_counts = take_wordCounts(read_coldata(df, col))
    print(word_counts)
    make_removedFile(word_counts.copy(), col)
    make_synonymFile(word_counts.copy(), col)