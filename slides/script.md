# Script báo cáo BTL — Urban Sound Classification

> Tổng thời gian dự kiến: **~17-20 phút**.
> Mỗi slide ~30-60 giây.

---

## Slide 1 — Title

Em chào thầy ạ, em là Ngô Thái Minh Tiến, mã số sinh viên 2252809. Hôm nay em xin được trình bày báo cáo Bài tập lớn môn Học Sâu với đề tài: **Phân loại âm thanh đô thị trên bộ dữ liệu UrbanSound8K sử dụng Log-Mel Spectrogram và các mô hình CNN, ResNet**. Giảng viên hướng dẫn của em là thầy Huỳnh Văn Thống. Bây giờ em xin phép bắt đầu phần trình bày ạ.

---

## Slide 2 — Nội dung trình bày

Báo cáo của em gồm 7 phần chính. Em sẽ trình bày lần lượt: bài toán và bộ dữ liệu, kết quả phân tích dữ liệu khám phá, tổng quan tài liệu, pipeline xử lý và các mô hình được sử dụng, thiết kế thực nghiệm, kết quả và phân tích, cuối cùng là kết luận.

---

## Slide 3 — Bài toán

Bài toán em giải quyết là **Urban Sound Classification** — phân loại âm thanh đô thị. Cho một đoạn âm thanh ngắn, mô hình cần dự đoán nó thuộc 1 trong 10 lớp âm thanh đã định trước. Đây là bài toán **supervised multi-class classification**.

Về ý nghĩa thực tế, một hệ thống như vậy có thể ứng dụng trong giám sát môi trường đô thị, thành phố thông minh, phát hiện các sự kiện bất thường như tiếng súng hay còi báo động, hỗ trợ phân tích giao thông, hoặc quan trắc tiếng ồn.

---

## Slide 4 — Bộ dữ liệu UrbanSound8K

Em sử dụng bộ dữ liệu **UrbanSound8K**, gồm 8732 audio clip thuộc 10 lớp, được chia sẵn thành 10 fold. Thời lượng trung bình mỗi clip khoảng 3.61 giây.

Một điểm cần lưu ý là dữ liệu gốc khá **không đồng nhất**: sample rate trải dài từ 8000 Hz tới 192000 Hz, và có cả file mono lẫn stereo. Đây là vấn đề em sẽ giải quyết ở phần tiền xử lý.

10 lớp âm thanh bao gồm: air_conditioner, car_horn, children_playing, dog_bark, drilling, engine_idling, gun_shot, jackhammer, siren và street_music.

---

## Slide 5 — Lý do chọn UrbanSound8K

Em chọn UrbanSound8K vì 4 lý do chính:

Thứ nhất, **dữ liệu đã được gán nhãn rõ ràng**, phù hợp với bài toán supervised classification.

Thứ hai, dataset **đã chia sẵn 10 fold**, rất thuận lợi để tổ chức train, validation, test.

Thứ ba, **kích thước vừa phải** — đủ để huấn luyện mô hình học sâu nhưng vẫn nằm trong phạm vi tài nguyên cá nhân, có thể chạy trên Colab.

Thứ tư, đây là **dataset chuẩn học thuật**, đã được nhiều nghiên cứu benchmark, giúp em dễ so sánh kết quả.

---

## Slide 6 — EDA

Em đã thực hiện phân tích dữ liệu khám phá và phát hiện ra 3 vấn đề lớn cần xử lý:

Thứ nhất, **sample rate không đồng nhất** từ 8 kHz tới 192 kHz. Nếu để nguyên, mô hình có thể học **format khác biệt** thay vì đặc trưng âm học.

Thứ hai, **mix mono và stereo** — số kênh khác nhau.

Thứ ba, **duration không cố định**, dao động quanh 3.61 giây.

Ngoài ra, em cũng nhận thấy các lớp có **mức độ khó khác nhau**. Các lớp như car_horn, gun_shot, siren có đặc trưng nổi bật, dễ phân biệt. Ngược lại, các âm thanh máy móc liên tục như air_conditioner, engine_idling, drilling, jackhammer rất giống nhau trong miền tần số — đây sẽ là **cụm khó** của bài toán.

---

## Slide 7 — Quyết định tiền xử lý

Từ kết quả EDA, em đưa ra 4 quyết định tiền xử lý:
- Chuyển toàn bộ về **mono** để đồng nhất số chiều.
- **Resample về 22050 Hz** — cân bằng giữa giữ thông tin và giảm chi phí tính toán.
- **Pad hoặc truncate về 4 giây**, tương ứng 88200 mẫu.
- **Trích xuất log-Mel spectrogram** với các tham số: n_fft = 1024, hop_length = 512, n_mels = 128, top_db = 80.

Đây là các giá trị tiêu chuẩn được sử dụng phổ biến trong các nghiên cứu audio classification.

---

## Slide 8 — Tổng quan tài liệu

Về tổng quan, có 2 hướng tiếp cận chính cho bài toán phân loại âm thanh.

**Hướng truyền thống** dùng đặc trưng thủ công như MFCC, zero-crossing rate, spectral centroid… kết hợp với các bộ phân loại như SVM, k-NN, Random Forest. Hạn chế là phụ thuộc mạnh vào chất lượng đặc trưng được chọn.

**Hướng Deep Learning** chuyển tín hiệu sang spectrogram — coi như ảnh 2 chiều — rồi dùng các kiến trúc CNN, ResNet. Đây là hướng em chọn vì mô hình có thể **tự học đặc trưng phân cấp** từ dữ liệu mà không cần thiết kế tay.

Ngoài ra, các kỹ thuật **augmentation như SpecAugment** đã chứng minh hiệu quả trong việc tăng khả năng tổng quát hóa của mô hình.

---

## Slide 9 — Pipeline tổng quan

Đây là pipeline tổng thể của project em. Bắt đầu từ file WAV gốc, qua bước preprocess gồm chuyển mono, resample, pad/truncate. Sau đó tính log-Mel spectrogram để được tensor có shape `[1 × 128 × 173]`. Tensor này đi qua một trong ba mô hình: CNN baseline, ResNet18, hoặc ResNet18 với augmentation. Cuối cùng softmax cho ra dự đoán 1 trong 10 lớp.

Toàn bộ pipeline được tổ chức thành các file Python riêng biệt trong thư mục `src/`, giúp dễ kiểm thử và dễ chạy lại.

---

## Slide 10 — CNN baseline

Mô hình đầu tiên em xây dựng là **CNN baseline**. Kiến trúc gồm 4 block convolutional, số channel tăng dần 16 → 32 → 64 → 128. Mỗi block có Conv 3×3, BatchNorm và ReLU. Cuối cùng là AdaptiveAvgPool và classifier gồm 2 lớp Fully Connected với Dropout 0.3.

Vai trò của CNN baseline là làm **mốc tham chiếu** — cho thấy với một kiến trúc đơn giản, pipeline đã chuẩn hóa tốt có thể đạt hiệu năng đến đâu, từ đó so sánh với các mô hình phức tạp hơn.

---

## Slide 11 — ResNet18

Mô hình thứ hai là **ResNet18**, được em tự cài từ đầu và **điều chỉnh** để phù hợp với đầu vào spectrogram.

Hai điều chỉnh chính:
- **Input 1 channel** thay vì 3 channel như ảnh RGB.
- **Stem** được điều chỉnh để giữ chi tiết cục bộ — không pool sớm như ResNet gốc dùng cho ảnh, vì các cấu trúc nhỏ trong spectrogram có thể mang ý nghĩa phân biệt quan trọng.

Mô hình có 4 stage, mỗi stage 2 BasicBlock với residual connection. Em **không dùng pretrained** để đánh giá thuần khả năng học từ dataset UrbanSound8K.

---

## Slide 12 — ResNet18 + Augmentation

Mô hình thứ ba là **ResNet18 kết hợp augmentation** trong miền spectrogram. Các phép biến đổi gồm:
- Thêm **Gaussian noise** nhẹ.
- **Time shift** — dịch theo trục thời gian.
- **Frequency masking** — che một dải tần số.
- **Time masking** — che một dải thời gian.

Lưu ý là các augmentation này **chỉ áp dụng trên tập train**, validation và test giữ nguyên để đảm bảo đánh giá khách quan.

Mục tiêu của augmentation là tăng tính đa dạng của dữ liệu huấn luyện, giúp mô hình **học đặc trưng bền vững hơn** và đặc biệt cải thiện hiệu năng ở các lớp khó.

---

## Slide 13 — Thiết kế thực nghiệm

Em thiết kế 3 cấu hình thực nghiệm để so sánh:
- **C1: CNN baseline** — không augmentation.
- **C2: ResNet18** — không augmentation.
- **C3: ResNet18 + augmentation**.

Để đảm bảo so sánh **công bằng**, em giữ cố định toàn bộ pipeline đầu vào: cùng dataset, cùng split fold, cùng log-Mel spectrogram. Sự khác biệt giữa các thí nghiệm chỉ đến từ **kiến trúc mô hình và chiến lược augmentation**.

Em đánh giá bằng 3 chỉ số: **Accuracy** cho hiệu năng tổng thể, **Macro-F1** cho cân bằng giữa các lớp, và **Confusion Matrix** để phân tích các cặp lớp dễ nhầm.

---

## Slide 14 — Quy trình train / eval

Quy trình huấn luyện của em dùng optimizer Adam hoặc AdamW, loss CrossEntropy với label smoothing, scheduler giảm learning rate khi val_loss đứng. Em có **early stopping** dựa trên val_acc để tránh overfit.

Toàn bộ quá trình huấn luyện được **log lên Weights & Biases** — bao gồm loss, accuracy, F1 mỗi epoch — giúp theo dõi real-time và đảm bảo tính tái lập.

Phần đánh giá: em load checkpoint tốt nhất theo val_acc, chạy trên tập test, xuất ra metrics JSON, classification report, confusion matrix dạng PNG và file predictions để phân tích sâu hơn.

---

## Slide 15 — Kết quả tổng quan

Đây là bảng kết quả của 3 mô hình trên tập test.

CNN baseline đạt **74.46%** Accuracy, **75.25%** Macro-F1.
ResNet18 đạt **72.97%** Accuracy, **72.01%** Macro-F1.
ResNet18 + augmentation đạt **72.97%** Accuracy nhưng **Macro-F1 tăng lên 76.17%**.

Có 3 quan sát chính:

Thứ nhất, **CNN baseline đạt Accuracy cao nhất** — cho thấy kiến trúc đơn giản phù hợp với dataset có kích thước vừa.

Thứ hai, **ResNet18 chưa vượt CNN** — kiến trúc sâu hơn không tự động tốt hơn nếu chưa được tối ưu kỹ.

Thứ ba và quan trọng nhất: **ResNet18 + augmentation tuy không tăng Accuracy nhưng Macro-F1 tăng rõ rệt** — chứng tỏ augmentation đã giúp mô hình **cân bằng hơn giữa các lớp**.

---

## Slide 16 — CNN baseline F1 theo lớp

Đi vào chi tiết từng lớp của CNN baseline.

Các **lớp dễ** như car_horn (F1 = 0.90), gun_shot (0.89), siren (0.88) đạt hiệu năng rất cao — vì chúng có đặc trưng âm học nổi bật, dễ phân biệt trên miền phổ.

Tuy nhiên, **lớp yếu nhất là air_conditioner** với F1 chỉ **0.31**. Đây là một âm thanh nền liên tục, dễ bị mô hình nhầm sang các lớp âm máy móc khác.

Đây cũng là **vấn đề cốt lõi** sẽ được thảo luận ở các slide tiếp theo.

---

## Slide 17 — ResNet18 F1 theo lớp

Với ResNet18, em quan sát một điểm rất đáng chú ý: **air_conditioner sụp đổ** xuống F1 chỉ **0.089** — thấp hơn rất nhiều so với CNN baseline.

Điều này cho thấy ResNet18, dù sâu hơn, lại bị **thiên về các đặc trưng mạnh** và bỏ qua các pattern yếu, mang tính nền như air_conditioner. Đây là lý do chính khiến Macro-F1 của ResNet18 thấp hơn CNN baseline.

Các lớp khác như siren, engine_idling, gun_shot vẫn đạt F1 cao, nhưng việc sụp đổ ở 1 lớp đã kéo Macro-F1 tổng thể đi xuống.

---

## Slide 18 — ResNet18 + Augmentation F1 theo lớp

Khi bổ sung augmentation, kết quả thay đổi rõ rệt. Quan trọng nhất: **air_conditioner tăng từ 0.089 lên 0.372** — cải thiện gần **0.28 điểm**.

Đây là **bằng chứng mạnh nhất** cho hiệu quả của augmentation: nó cứu được lớp khó nhất của bài toán.

Các lớp khác như car_horn, gun_shot, dog_bark cũng tăng. Tuy nhiên một số lớp giảm nhẹ — ví dụ engine_idling giảm từ 0.87 xuống 0.71. Điều này cho thấy augmentation tạo ra một **sự đánh đổi**: mô hình cân bằng hơn nhưng mất đi độ sắc bén ở một vài lớp mạnh.

---

## Slide 19 — Phân tích confusion matrix

Phân tích confusion matrix cho thấy các cặp nhầm lẫn chính đúng như em đã dự đoán từ phần EDA:

- **air_conditioner ↔ engine_idling** — cả hai là âm nền liên tục, phổ tương tự.
- **drilling ↔ jackhammer** — đều là âm cơ khí với dải tần mạnh.
- Các âm môi trường yếu thường bị nhầm sang lớp machinery.

**Augmentation giúp giảm các nhầm lẫn dễ overfit**, nhưng **chưa giải quyết hoàn toàn** confusion giữa cụm machinery class — đây là giới hạn cấu trúc do log-Mel spectrogram của các âm này quá giống nhau.

---

## Slide 20 — So sánh trực tiếp

Nếu so sánh tổng thể:
- Theo **Accuracy**, CNN baseline là tốt nhất.
- Theo **Macro-F1**, ResNet18 + augmentation là tốt nhất.

Bài học rút ra là **không nên chỉ dùng một chỉ số duy nhất** để kết luận. Một mô hình có Accuracy cao chưa chắc đã hoạt động tốt ở mọi lớp. Trong bài toán urban sound classification với class không đồng đều về độ khó, **Macro-F1 phản ánh chất lượng thực tế tốt hơn**.

---

## Slide 21 — Hạn chế của đề tài

Em ý thức được một số hạn chế của đề tài:

- **Chỉ dùng 1 fold split** thay vì 10-fold cross-validation chuẩn của UrbanSound8K, do giới hạn về compute trên Colab.
- **Chưa khảo sát sâu** các chiến lược tuning cho ResNet18.
- **Chưa thử các kỹ thuật nâng cao** như Mixup, OneCycleLR, ensemble.
- Phần **literature review** mới dừng ở mức tổng quan.

Tuy nhiên, trong phạm vi một BTL cá nhân, các kết quả hiện có vẫn đủ để rút ra **insight có giá trị về hiệu năng tương đối** giữa các phương án.

---

## Slide 22 — Kết luận

Tổng kết lại, em đã thực hiện được:
- Xây dựng **pipeline hoàn chỉnh** từ EDA tới preprocess, train, evaluate.
- So sánh **công bằng** 3 mô hình trên cùng setup.
- Phân tích **định lượng và định tính** với per-class F1 và confusion matrix.
- Log đầy đủ qua W&B để đảm bảo tái lập.

Ba insight chính em rút ra:
1. **Kiến trúc lớn hơn không tự động tốt hơn** trên dataset vừa.
2. **Augmentation** quan trọng cho cân bằng lớp, không nhất thiết tăng Accuracy.
3. **Cụm machinery** — air_conditioner, engine_idling, drilling, jackhammer — là **bottleneck** của bài toán, do log-Mel spectrogram của chúng quá giống nhau.

---

## Slide 23 — Tài liệu tham khảo

Đây là các tài liệu chính em đã tham khảo trong quá trình thực hiện đề tài, bao gồm paper gốc của UrbanSound8K do Salamon và cộng sự công bố, ResNet của He et al., SpecAugment của Park et al., cùng các công trình liên quan đến CNN cho audio classification.

---

## Slide 24 — Cảm ơn

Đó là toàn bộ phần báo cáo của em. Toàn bộ source code đã được public trên GitHub, và quá trình huấn luyện được log đầy đủ trên Weights & Biases — thầy có thể tra cứu bất kỳ lúc nào.

Em cảm ơn thầy đã lắng nghe, và rất mong nhận được góp ý từ thầy để hoàn thiện đề tài tốt hơn ạ.

---

# Phụ lục: Trả lời các câu hỏi thầy có thể hỏi

### Q1: Vì sao chọn log-Mel mà không phải MFCC?
A: Log-Mel giữ được nhiều thông tin tần số hơn MFCC (vốn là biến đổi DCT của log-Mel để giảm chiều). Với CNN/ResNet vốn xử lý spectrogram như ảnh 2D, log-Mel phù hợp hơn vì giữ cấu trúc không gian thời gian–tần số nguyên vẹn. MFCC thường dùng cho mô hình truyền thống với feature có chiều thấp.

### Q2: Vì sao không dùng pretrained ResNet18 từ ImageNet?
A: Hai lý do. Thứ nhất, ImageNet được train trên ảnh tự nhiên (RGB) — đặc trưng học được không hoàn toàn phù hợp với log-Mel spectrogram (1 channel, ngữ nghĩa khác). Thứ hai, em muốn đánh giá **thuần khả năng học** của ResNet18 từ UrbanSound8K, không bị "lẫn" hiệu quả của pretrained. Nếu mở rộng đề tài, em sẽ thử pretrained như một baseline transfer learning.

### Q3: Vì sao ResNet18 lại tệ hơn CNN ở lớp air_conditioner?
A: ResNet18 có nhiều tham số hơn CNN baseline. Với dataset có ~7000 sample train, ResNet18 dễ overfit và **học mạnh các đặc trưng đặc trưng**, bỏ qua các pattern yếu/mờ như âm nền của air_conditioner. CNN baseline đơn giản hơn nên ít bias về các đặc trưng dominant.

### Q4: Macro-F1 là gì và vì sao quan trọng?
A: Macro-F1 là trung bình F1-score của từng lớp, **không trọng số theo số lượng sample**. Khi các lớp có độ khó khác nhau (như UrbanSound8K), Macro-F1 phản ánh **mô hình có hoạt động đều giữa các lớp không**. Một mô hình Accuracy cao nhưng Macro-F1 thấp nghĩa là nó chỉ giỏi ở một số lớp dễ và yếu ở các lớp khó — không phải mô hình tốt thực sự.

### Q5: Tại sao 4 giây, sao không 3 giây hoặc 5 giây?
A: 4 giây gần với percentile cao của duration trong dataset (trung bình 3.61s, max 4s do cách dataset được sample). Chọn 4s giúp **giữ nguyên hầu hết file** mà không cần truncate nhiều. Với 22050 Hz, 4s = 88200 sample, một số tròn dễ xử lý.

### Q6: Tại sao chỉ 1 fold split mà không phải 10-fold CV?
A: Đây là **hạn chế thực sự** của đề tài. Lý do là giới hạn về compute Colab Pro — mỗi run mất ~30 phút, chạy 10 fold × 3 model = 30 run = ~15 giờ compute. Em chấp nhận trade-off này để có thể chạy được trên hạ tầng cá nhân. Trong kết quả em báo cáo so sánh tương đối giữa 3 model trên cùng split — vẫn có ý nghĩa.

### Q7: Augmentation nào trong 4 phép đóng góp nhiều nhất?
A: Em chưa làm ablation chi tiết. Theo paper SpecAugment, frequency masking và time masking đóng góp lớn nhất. Em đoán Gaussian noise và time shift hỗ trợ nhưng đóng góp nhỏ hơn. Đây là hướng mở rộng đề tài.

### Q8: Tại sao Test Accuracy của ResNet18 và ResNet18+aug bằng nhau?
A: Đây là một quan sát thú vị. Augmentation **redistribute hiệu năng giữa các lớp** thay vì tăng Accuracy tổng thể. Cụ thể, nó cứu air_conditioner (+0.28 F1) nhưng làm giảm một số lớp đã mạnh (engine_idling −0.15). Kết quả số mẫu predict đúng tổng cộng không đổi, nhưng **phân bố** thay đổi — thể hiện qua Macro-F1.

### Q9: Project có gì khác biệt so với paper UrbanSound8K gốc?
A: Em không claim đề tài mới. Mục tiêu là **triển khai pipeline đầy đủ** và **so sánh công bằng các phương án** trong phạm vi BTL. Paper gốc dùng SB-CNN đạt ~73% với 10-fold CV. Em dùng kiến trúc tương tự (CNN baseline), đạt 74% trên 1 fold — nằm trong khoảng kỳ vọng.

### Q10: Nếu có thêm thời gian, em sẽ làm gì?
A: Có 3 hướng. Một, chạy 10-fold CV để báo cáo mean ± std chuẩn paper. Hai, thử các kỹ thuật như Mixup, OneCycleLR, ensemble. Ba, thử pretrained ResNet hoặc PANNs để xem hiệu quả transfer learning.
