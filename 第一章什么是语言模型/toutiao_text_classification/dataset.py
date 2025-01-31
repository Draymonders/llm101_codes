import torch
from torch.utils.data import Dataset

LABEL_MAP = {
    "100": 0,  #  民生 故事 news_story
    "101": 1,  #  文化 文化 news_culture
    "102": 2,  #  娱乐 娱乐 news_entertainment
    "103": 3,  #  体育 体育 news_sports
    "104": 4,  #  财经 财经 news_finance
    "106": 5,  #  房产 房产 news_house
    "107": 6,  #  汽车 汽车 news_car
    "108": 7,  #  教育 教育 news_edu
    "109": 8,  #  科技 科技 news_tech
    "110": 9,  #  军事 军事 news_military
    "112": 10,  #  旅游 旅游 news_travel
    "113": 11,  #  国际 国际 news_world
    "114": 12,  #  证券 股票 stock
    "115": 13,  #  农业 三农 news_agriculture
    "116": 14,  #  电竞 游戏 news_game
}


class NewsDataset(Dataset):
    def __init__(self, filename, tokenizer, max_len=128, toy=True):
        self.data = []
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                item_id, cid, cname, title, keywords = line.strip().split("_!_")
                self.data.append({"title": title, "label": LABEL_MAP[cid]})
            if toy:
                self.data = self.data[:100*len(LABEL_MAP)]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):  # <X, y>
        title = self.data[idx]["title"]
        label = self.data[idx]["label"]

        # text encoding
        encoding = self.tokenizer(
            title,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(label, dtype=torch.long),
        }
