I was reading about data locality , tried to simulate a small read time delta with 1k reads 
---
# 10k users

    Normalized time: 2.100825548171997
    Normalized time with JOINS : 2.606114149093628
    Document time: 0.05394482612609863

Now I added an index ( reduce time complexity )

    Normalized time: 0.14394330978393555
    Normalized time with JOINS : 0.04903388023376465
    Document time: 0.05486655235290527

---

# 1 lakh users
    
    Normalized time: 16.939746856689453
    Normalized time with JOINS : 19.45433783531189
    Document time: 0.10438704490661621

Now I added an index ( reduce time complexity )

    Normalized time: 0.18265223503112793
    Normalized time with JOINS : 0.07358741760253906
    Document time: 0.06902194023132324


So indexes do reduce the search space a lot as seen in last experiment. 

Document DBs scale pretty well for this usecase with data volume . 

---
# 10 lakh users

    Normalized time: 185.72315454483032
    Normalized time with JOINS : 170.112783908844
    Document time: 0.034841060638427734

Now I added an index ( reduce time complexity )

    Normalized time: 0.09443140029907227
    Normalized time with JOINS : 0.042998313903808594
    Document time: 0.044394493103027344
