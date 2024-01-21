true_heads = ["0.1", "3.0", "0.10", "2.2", "4.11", "5.5", "6.9", "5.8", "5.9", "7.3", "7.9", "8.6", "8.10", "10.7", "11.10", "9.9", "9.6", "10.0", "9.0", "9.7", "10.1", "10.2", "10.6", "10.10", "11.2", "11.9"]
true_heads = set(true_heads)

def check_list(l):
    l = set(l)
    tp = 0
    fp = 0
    for i in l:
        if i in true_heads:
            tp += 1
        else:
            fp += 1
    print(f"proportions: tp: {tp/len(true_heads)}, fp: {fp/len(l)}")
    print(f"tp: {tp}, fp: {fp}, total: {len(l)}, true total: {len(true_heads)}")
    return tp, fp

l = ["0.1","0.10","3.0","5.5","7.9","8.6","8.10","9.6","9.8","9.9","10.0","10.1","10.2","10.3","10.10","11.2"]


print(check_list(l))