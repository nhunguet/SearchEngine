class BlogPost:
    def __init__(self, title, content):
        self.title = title
        self.content = content
        self.keywords = []
        self.meta_description = ""
        self.headers = []

    def add_keyword(self, keyword):
        self.keywords.append(keyword)
    
    def optimize_meta(self, meta_description):
        self.meta_description = meta_description
    
    def optimize_headers(self, headers):
        self.headers = headers
    
    def display_optimized_content(self):
        print(f"Title: {self.title}")
        print(f"Meta Description: {self.meta_description}")
        print("\nContent:")
        for header, text in self.content.items():
            print(f"{header}: {text}")

# Tạo bài viết blog mẫu
post = BlogPost(
    title="10 Delicious and Nutritious Healthy Breakfast Ideas",
    content={
        "Smoothie Bowl Recipes": "Include various fruit combinations...",
        "High-Protein Breakfasts": "Try scrambled eggs with spinach..."
    }
)

# Thêm từ khóa
post.add_keyword("healthy breakfast ideas")
post.add_keyword("quick healthy breakfast recipes")
post.add_keyword("nutritious morning meals")

# Tối ưu hóa mô tả meta
post.optimize_meta("Explore a variety of quick and nutritious breakfast ideas that will kickstart your day.")

# Tối ưu tiêu đề phụ
post.optimize_headers(["Smoothie Bowl Recipes", "High-Protein Breakfasts"])

# Hiển thị bài viết đã tối ưu
post.display_optimized_content()
