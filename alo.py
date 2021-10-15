import numpy as np
arr = np.array( [[ 4, 2, 2],
                 [ 4, 3, 4]] )
 
# In ra loại của đối tượng mảng vừa được tạo
print("Array thuộc loại: ", type(arr))
 
# In ra số chiều (trục)
print("Số chiều: ", arr.ndim)
 
# In ra hình dạng của mảng 
print("Dạng của mảng: ", arr.shape)
 
# In ra tổng số phần tử
print("Tổng số phần tử: ", arr.size)
 
# In ra loại dữ liệu của phần tử trong mảng
print("Array chứa các phần tử kiểu: ", arr.dtype)
compare = y_test_label==y_pred
print(y_test_label.shape)
print(y_pred.shape)
print(compare.shape)
if not os.path.exists('imger'):
    os.mkdir('imger')
for i, n in enumerate(compare):
    
    if str(n) == 'False':
        print(i)
        px = X_test[i]# đây là ảnh sai
        cv2.imwrite('imger/'+str(i)+'.jpg', px)
        cv2.waitKey(1)