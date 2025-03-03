# Danh sách các trang web thực tế với điểm số giả lập
webpages = [
    {"name": "Wikipedia", "relevance": 9.5, "authority": 10.0, "engagement": 7.5},
    {"name": "Google Blog", "relevance": 9.0, "authority": 9.5, "engagement": 8.5},
    {"name": "TechCrunch", "relevance": 8.5, "authority": 8.5, "engagement": 9.0},
    {"name": "Stack Overflow", "relevance": 9.0, "authority": 9.0, "engagement": 9.5},
    {"name": "Medium", "relevance": 8.0, "authority": 7.5, "engagement": 8.5},
    {"name": "Reddit", "relevance": 7.5, "authority": 7.0, "engagement": 9.5},
    {"name": "Quora", "relevance": 7.0, "authority": 7.5, "engagement": 8.0},
]

# Trọng số của từng yếu tố trong thuật toán xếp hạng
weights = {
    "relevance": 0.4,  # Độ quan trọng của từ khóa (40%)
    "authority": 0.3,  # Độ uy tín của trang web (30%)
    "engagement": 0.3  # Mức độ tương tác của người dùng (30%)
}

# Tính điểm xếp hạng cho từng trang
for page in webpages:
    page["ranking_score"] = (
        weights["relevance"] * page["relevance"] +
        weights["authority"] * page["authority"] +
        weights["engagement"] * page["engagement"]
    )

# Sắp xếp các trang theo điểm xếp hạng giảm dần
ranked_pages = sorted(webpages, key=lambda x: x["ranking_score"], reverse=True)

# Hiển thị kết quả
print("Ranking of Real Websites:")
for i, page in enumerate(ranked_pages, start=1):
    print(f"{i}. {page['name']} - Score: {page['ranking_score']:.2f}")
