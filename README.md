# Tìm hiểu mạng tích chập CNN

Trong vài thập kỷ qua, Deep Learning đã chứng tỏ là một công cụ rất mạnh mẽ vì khả năng xử lý lượng lớn dữ liệu. Mối quan tâm sử dụng các lớp ẩn đã vượt qua các kỹ thuật truyền thống, đặc biệt là trong nhận dạng mẫu. Một trong những mạng nơ-ron sâu phổ biến nhất là Mạng nơ-ron chuyển đổi (còn được gọi là CNN hoặc ConvNet) trong học sâu, đặc biệt là khi nói đến các ứng dụng Thị giác máy tính.

![image](https://github.com/thangdtph27626/CNN/assets/109157942/d7b563a5-2c21-4fb2-9afa-624d6fd5caaf)

## Bối cảnh của CNN

CNN lần đầu tiên được phát triển và sử dụng vào khoảng những năm 1980. Điều tốt nhất mà CNN có thể làm vào thời điểm đó là nhận dạng các chữ số viết tay. Nó chủ yếu được sử dụng trong lĩnh vực bưu chính để đọc mã zip, mã pin, v.v. Điều quan trọng cần nhớ về bất kỳ mô hình học sâu là nó yêu cầu một lượng lớn dữ liệu để đào tạo và cũng cần nhiều tài nguyên máy tính. Đây là một nhược điểm lớn đối với CNN vào thời kỳ đó và do đó CNN chỉ giới hạn trong lĩnh vực bưu chính và không thể thâm nhập vào thế giới của học máy.

## CNN là gì?
Trong học sâu, mạng nơ-ron tích chập ( CNN/ConvNet ) là một lớp mạng nơ-ron sâu, được áp dụng phổ biến nhất để phân tích hình ảnh trực quan. Bây giờ khi nghĩ đến mạng nơ-ron, chúng ta nghĩ đến phép nhân ma trận nhưng ConvNet thì không như vậy. Nó sử dụng một kỹ thuật đặc biệt gọi là Convolution. Trong toán học tích chập là một phép toán trên hai hàm tạo ra hàm thứ ba biểu thị hình dạng của hàm này được biến đổi bởi hàm kia như thế nào.

![image](https://github.com/thangdtph27626/CNN/assets/109157942/2f7e83a5-697d-4e37-b9b8-586c7daf35f8)

Nhưng chúng ta không thực sự cần phải đi sâu vào phần toán học để hiểu CNN là gì hoặc nó hoạt động như thế nào.

Điểm mấu chốt là vai trò của ConvNet là giảm hình ảnh thành dạng dễ xử lý hơn mà không làm mất các tính năng quan trọng để có được dự đoán tốt.

## Làm thế nào nó hoạt động?

Trước khi bắt đầu làm việc với CNN, chúng ta hãy tìm hiểu những điều cơ bản như hình ảnh là gì và nó được thể hiện như thế nào. Hình ảnh RGB không là gì ngoài ma trận các giá trị pixel có ba mặt phẳng trong khi hình ảnh thang độ xám giống nhau nhưng nó có một mặt phẳng duy nhất. Hãy nhìn vào hình ảnh này để hiểu thêm.

![image](https://github.com/thangdtph27626/CNN/assets/109157942/b31e3e12-ad58-430b-b324-1ae578074928)

Để đơn giản, chúng ta hãy tập trung vào các hình ảnh thang độ xám khi chúng ta cố gắng hiểu cách thức hoạt động của CNN.

![image](https://github.com/thangdtph27626/CNN/assets/109157942/0e567df8-9cce-448c-9b55-af31bb41952c)

Hình ảnh trên cho thấy tích chập là gì. Chúng ta lấy bộ lọc/hạt nhân (ma trận 3 × 3) và áp dụng nó cho hình ảnh đầu vào để có được tính năng tích chập. Tính năng phức tạp này được chuyển sang lớp tiếp theo.

<figure class="wp-block-image"><img decoding="async" src="https://editor.analyticsvidhya.com/uploads/419681_GcI7G-JLAQiEoCON7xFbhg.gif" alt="Quá trình tích chập trong Mạng thần kinh tích chập"></figure>

Trong trường hợp màu RGB, kênh hãy xem hoạt ảnh này để hiểu hoạt động của nó

<figure class="wp-block-image"><img decoding="async" src="https://editor.analyticsvidhya.com/uploads/556091_ciDgQEjViWLnCbmX-EeSrA.gif" alt="Hoạt động của ConvNet trên ảnh màu RGB"></figure>

Mạng nơ-ron tích chập bao gồm nhiều lớp nơ-ron nhân tạo. Tế bào thần kinh nhân tạo, mô phỏng sơ bộ các tế bào thần kinh sinh học của chúng, là các hàm toán học tính toán tổng trọng số của nhiều đầu vào và đưa ra giá trị kích hoạt. Khi bạn nhập một hình ảnh vào ConvNet, mỗi lớp sẽ tạo ra một số chức năng kích hoạt được chuyển cho lớp tiếp theo.

Lớp đầu tiên thường trích xuất các đặc điểm cơ bản như các cạnh ngang hoặc chéo. Đầu ra này được chuyển sang lớp tiếp theo để phát hiện các tính năng phức tạp hơn như các góc hoặc các cạnh tổ hợp. Khi chúng ta tiến sâu hơn vào mạng, nó có thể xác định các tính năng phức tạp hơn nữa như vật thể, khuôn mặt,...

![image](https://github.com/thangdtph27626/CNN/assets/109157942/746128e3-322b-4a7a-87ba-1f025a193691)

Dựa trên bản đồ kích hoạt của lớp tích chập cuối cùng, lớp phân loại đưa ra một tập hợp điểm tin cậy (giá trị từ 0 đến 1) xác định khả năng hình ảnh thuộc về một “lớp”. Ví dụ: nếu bạn có ConvNet phát hiện mèo, chó và ngựa thì đầu ra của lớp cuối cùng là khả năng hình ảnh đầu vào có chứa bất kỳ động vật nào trong số đó.

![image](https://github.com/thangdtph27626/CNN/assets/109157942/9b51178b-ea54-45cc-a979-fabf1c8d6a62)

## Lớp gộp là gì?
Tương tự như Lớp Convolutional, lớp Pooling chịu trách nhiệm giảm kích thước không gian của Tính năng Convolved. Điều này nhằm giảm sức mạnh tính toán cần thiết để xử lý dữ liệu bằng cách giảm kích thước. Có hai loại gộp chung là gộp trung bình và gộp gộp tối đa. Tôi mới chỉ có kinh nghiệm với Max Pooling cho đến nay và chưa gặp bất kỳ khó khăn nào.
<figure class="wp-block-image"><img decoding="async" src="https://editor.analyticsvidhya.com/uploads/254781_uoWYsCV5vBU8SHFPAPao-w.gif" alt="Lớp gộp làm giảm sức mạnh tính toán cần thiết để xử lý dữ liệu trong CNN."></figure>

Vì vậy, những gì chúng tôi làm trong Max Pooling là chúng tôi tìm giá trị tối đa của pixel từ một phần hình ảnh được bao phủ bởi hạt nhân. Max Pooling cũng hoạt động như một Bộ giảm tiếng ồn . Nó loại bỏ hoàn toàn các kích hoạt nhiễu và cũng thực hiện khử nhiễu cùng với việc giảm kích thước.

Mặt khác, Average Pooling trả về giá trị trung bình của tất cả các giá trị từ phần hình ảnh được Kernel bao phủ. Tổng hợp trung bình chỉ đơn giản thực hiện giảm kích thước như một cơ chế khử nhiễu. Do đó, chúng ta có thể nói rằng Max Pooling hoạt động tốt hơn rất nhiều so với Average Pooling

<figure class="wp-block-image"><img decoding="async" src="https://editor.analyticsvidhya.com/uploads/597371_KQIEqhxzICU7thjaQBfPBQ.png" alt="Tổng hợp tối đa và tổng hợp trung bình trong ConvNet"></figure>

## Hạn chế của CNN
Bất chấp sức mạnh và sự phức tạp về tài nguyên của CNN, chúng vẫn cung cấp kết quả chuyên sâu. Về căn bản, nó chỉ là việc nhận ra những khuôn mẫu và chi tiết rất nhỏ và kín đáo đến mức mắt người không thể nhận ra. Nhưng khi hiểu được nội dung của một hình ảnh thì lại thất bại.

Chúng ta hãy xem ví dụ này. Khi chúng tôi chuyển hình ảnh bên dưới cho CNN, nó sẽ phát hiện một người ở độ tuổi giữa 30 và một đứa trẻ có lẽ khoảng 10 tuổi. Nhưng khi nhìn vào cùng một hình ảnh, chúng ta bắt đầu nghĩ đến nhiều tình huống khác nhau. Có thể đó là một ngày đi chơi của hai cha con, một chuyến dã ngoại hoặc có thể họ đang đi cắm trại. Có lẽ đó là sân trường và đứa trẻ ghi bàn, bố nó vui nên bế nó lên.

Những hạn chế này được thể hiện rõ ràng hơn khi áp dụng vào thực tế. Ví dụ: CNN được sử dụng rộng rãi để kiểm duyệt nội dung trên mạng xã hội. Nhưng mặc dù có nguồn tài nguyên hình ảnh và video khổng lồ mà họ đã được đào tạo về nó vẫn không thể chặn và xóa hoàn toàn nội dung không phù hợp. Hóa ra nó đã gắn cờ một bức tượng khỏa thân 30.000 năm tuổi trên Facebook.

Một số nghiên cứu đã chỉ ra rằng các CNN được đào tạo trên ImageNet và các bộ dữ liệu phổ biến khác không phát hiện được vật thể khi họ nhìn thấy chúng trong các điều kiện ánh sáng khác nhau và từ các góc độ mới.

Phải chăng điều này có nghĩa là CNN vô dụng? Tuy nhiên, bất chấp những hạn chế của mạng lưới thần kinh tích chập, không thể phủ nhận rằng chúng đã gây ra một cuộc cách mạng về trí tuệ nhân tạo. Ngày nay, CNN được sử dụng trong nhiều ứng dụng thị giác máy tính như nhận dạng khuôn mặt, tìm kiếm và chỉnh sửa hình ảnh, thực tế tăng cường, v.v. Như những tiến bộ trong ConvNets cho thấy, những thành tựu của chúng tôi rất đáng chú ý và hữu ích, nhưng chúng tôi vẫn còn rất xa mới có thể tái tạo được các thành phần chính của trí thông minh con người .

