import matplotlib.pyplot as plt

# tiny-imagenet-200 데이터셋의 정확도와 ECE 값
methods = [
    ("Single Network", 83.37, 5.63),
    ("Uniform", 62.90, 17.88),
    ("Greedy (ACC)", 83.07, 6.29),
    # ("Greedy (ECE)", 76.33, 5.03),
    ("Branch Soup", 84.92, 4.48),
]


# methods = [
#     ("Single Network", 90.19, 3.50),
#     # ("Ensemble", 90.60, 2.44),
#     ("Uniform", 73.80, 6.62),
#     ("Greedy (ACC)", 90.47, 3.61),
#     ("Greedy (ECE)", 90.50, 3.20),
#     ("Branch Soup", 90.27, 1.35),
# ]


# 시각화 스타일: {이름: (표시 라벨, 색상, 마커)}
style_map = {
    "Branch Soup": ("Branch Soup", 'purple', '*'),
    "Uniform": ("Uniform Soup", 'blue', 'o'),
    "Greedy (ACC)": ("Greedy Soup", 'blue', 's'),
    # "Greedy (ECE)": ("Greedy Soup (ECE)", 'green', 's'),
    "Single Network": ("Single Network", 'gray', 'D'),
}

# 그래프 초기화
plt.figure(figsize=(7, 6))

# 각 점 플로팅
for name, acc, ece in methods:
    label, color, marker = style_map[name]
    plt.scatter(ece, acc, label=label, color=color, marker=marker, s=165, zorder=3)

# 축 및 스타일 설정
plt.xlabel("ECE(↓) (%)", fontsize=16)
plt.ylabel("Accuracy(↑) (%)", fontsize=16)
# plt.title("Tiny-ImageNet-200: Accuracy vs -ECE", fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True)

plt.legend(loc='upper right', fontsize=13, frameon=True)


plt.xlim(3, 19)  # x축: -ECE
plt.ylim(62, 86)      # y축: Accuracy

# plt.xlim(3.5, 8.5)  # x축: -ECE
# plt.ylim(79, 85)      # y축: Accuracy

# plt.xlim(-7, -1)
# plt.ylim(72, 92)

# 이미지 저장
plt.savefig("accuracy_vs_ece_customized.png", dpi=300)
# plt.savefig("accuracy_vs_ece_customized.pdf")

plt.show()
