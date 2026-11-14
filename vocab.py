"""
공유 어휘: 서버 시작 시 set_classes() 로 한 번에 설정할 생활용품 이름 (영어, 소문자)

- 시각장애인이 자주 찾는 물건을 맨 앞에 두었다.
- SYNONYM_GROUPS: 모델이 비슷한 물건을 다른 이름으로 인식해도 놓치지 않도록,
  요청한 이름이 그룹에 있으면 같은 그룹의 이름들도 함께 찾는다.
  (예: "cup"을 요청했는데 모델이 "mug"로 인식한 경우에도 결과에 포함)
"""

# ---- 필수: 시각장애인이 자주 찾는 물건 ------------------------------------
ESSENTIALS = [
    "pill bottle", "medicine bottle", "pill box", "medicine", "eye drops", "inhaler",
    "key", "keychain",
    "remote control", "tv remote",
    "charger", "phone charger", "charging cable", "power bank", "earphones", "headphones",
    "glasses", "eyeglasses", "sunglasses", "glasses case",
    "wallet", "purse", "coin purse", "card", "credit card",
    "cell phone", "smartphone",
    "cup", "mug", "coffee cup", "paper cup", "tumbler", "glass", "water bottle", "bottle",
    "white cane", "watch", "hearing aid",
]

# ---- 주방·식사 ------------------------------------------------------------
KITCHEN = [
    "plate", "bowl", "spoon", "fork", "knife", "chopsticks", "cutting board", "frying pan", "pot",
    "kettle", "electric kettle", "teapot", "thermos", "lunch box", "food container", "lid",
    "can", "soda can", "beer can", "juice box", "milk carton", "wine glass", "straw", "napkin",
    "paper towel", "dish soap", "sponge", "rice cooker", "microwave", "refrigerator", "toaster",
    "coffee machine", "blender", "ladle", "spatula", "tray", "salt shaker", "sauce bottle",
    "ketchup", "jar", "snack bag", "bread", "banana", "apple", "orange", "egg", "instant noodles",
]

# ---- 개인용품·위생 --------------------------------------------------------
PERSONAL = [
    "toothbrush", "toothpaste", "towel", "soap", "hand sanitizer", "shampoo", "lotion",
    "comb", "hairbrush", "razor", "nail clipper", "tissue box", "tissue", "wet wipes", "face mask",
    "lip balm", "cosmetics", "mirror", "hair dryer", "cotton swab", "band-aid", "thermometer",
    "blood pressure monitor", "glucose meter",
]

# ---- 전자기기 -------------------------------------------------------------
ELECTRONICS = [
    "laptop", "tablet", "keyboard", "mouse", "monitor", "television", "speaker", "smart speaker",
    "radio", "camera", "game controller", "usb drive", "power strip", "extension cord", "plug",
    "light switch", "flashlight", "battery", "calculator", "clock", "alarm clock", "printer",
    "fan", "air conditioner remote", "electric fan", "lamp", "desk lamp", "light bulb",
]

# ---- 옷·외출 --------------------------------------------------------------
CLOTHING = [
    "hat", "cap", "scarf", "gloves", "jacket", "coat", "shirt", "pants", "socks", "shoes",
    "sneakers", "slippers", "sandals", "belt", "tie", "umbrella", "bag", "backpack", "handbag",
    "shopping bag", "plastic bag", "paper bag", "suitcase", "transit card", "id card",
]

# ---- 집안·가구 ------------------------------------------------------------
HOME = [
    "chair", "table", "desk", "sofa", "bed", "pillow", "blanket", "cushion", "door", "door handle",
    "doorknob", "window", "curtain", "drawer", "cabinet", "shelf", "trash can", "recycling bin",
    "basket", "box", "cardboard box", "hanger", "clothes hanger", "laundry basket", "vase",
    "plant", "flower pot", "picture frame", "candle", "toilet paper", "toilet", "sink", "faucet",
    "bathtub", "washing machine", "vacuum cleaner", "broom", "mop", "bucket", "stairs",
]

# ---- 문구·기타 ------------------------------------------------------------
STATIONERY_ETC = [
    "book", "notebook", "paper", "envelope", "pen", "pencil", "marker", "scissors", "tape",
    "stapler", "ruler", "folder", "sticky note", "braille book", "magnifier", "newspaper",
    "letter", "document", "receipt", "coin", "banknote", "toy", "ball", "teddy bear",
    "dog leash", "pet bowl", "cat", "dog", "person",
]

BASE_VOCAB: list[str] = []
for _group in (ESSENTIALS, KITCHEN, PERSONAL, ELECTRONICS, CLOTHING, HOME, STATIONERY_ETC):
    for _w in _group:
        if _w not in BASE_VOCAB:
            BASE_VOCAB.append(_w)

# ---- 배경 물체 ------------------------------------------------------------
# targets 없이 /detect 를 부를 때(전체 탐지) 결과에서 빼는 이름.
# 가구·건물 구조·사람처럼 거의 항상 보이고 손으로 집을 물건이 아닌 것들.
BACKGROUND: set[str] = {
    "table", "desk", "chair", "sofa", "bed", "wall", "floor", "ceiling", "person",
    "window", "door", "curtain", "cabinet", "shelf", "drawer", "stairs",
    "refrigerator", "washing machine", "toilet", "sink", "bathtub", "faucet",
}

# ---- 동의어 그룹 ----------------------------------------------------------
SYNONYM_GROUPS: list[list[str]] = [
    ["cup", "mug", "coffee cup", "paper cup", "tumbler", "glass"],
    ["water bottle", "bottle", "tumbler", "thermos"],
    ["pill bottle", "medicine bottle", "medicine", "pill box"],
    ["key", "keychain"],
    ["remote control", "tv remote", "air conditioner remote"],
    ["charger", "phone charger", "charging cable", "power bank"],
    ["glasses", "eyeglasses", "sunglasses"],
    ["wallet", "purse", "coin purse"],
    ["card", "credit card", "transit card", "id card"],
    ["cell phone", "smartphone"],
    ["earphones", "headphones"],
    ["trash can", "recycling bin"],
    ["door handle", "doorknob"],
    ["tissue", "tissue box"],
    ["bag", "handbag", "backpack", "shopping bag", "plastic bag", "paper bag"],
    ["shoes", "sneakers", "slippers", "sandals"],
    ["lamp", "desk lamp"],
    ["fan", "electric fan"],
    ["television", "monitor"],
]

# 흔히 쓰는 다른 표현 → 어휘에 있는 이름 (요청 단어를 바꿔서 찾음)
ALIASES: dict[str, str] = {
    "phone": "cell phone",
    "mobile phone": "cell phone",
    "cellphone": "cell phone",
    "iphone": "smartphone",
    "remote": "remote control",
    "keys": "key",
    "spectacles": "glasses",
    "pills": "pill bottle",
    "pill": "pill bottle",
    "tv": "television",
    "mobile": "cell phone",
    "earbuds": "earphones",
    "airpods": "earphones",
    "coffee mug": "mug",
    "drinking glass": "glass",
    "trash": "trash can",
    "garbage can": "trash can",
    "cable": "charging cable",
}

if __name__ == "__main__":
    print(f"기본 어휘 {len(BASE_VOCAB)}개")
