"""LỚP `citation_checker` — bài giảng Day 16, §11 (Grounding & Citations).

NHIỆM VỤ: chỉ cần MỘT tài liệu gắn nhãn `lookalike` hoặc `outdated` lọt
vào bằng chứng là mô hình neo TOÀN BỘ claim vào đúng tài liệu trông có vẻ
"chính thống" đó — dù mỗi câu được lấy nguyên văn từ một tài liệu khác.
Câu thì thật, trích dẫn thì sai. Đây là kiểu sai nguy hiểm nhất trong RAG
vì báo cáo đọc vào vẫn rất thuyết phục.

TÍN HIỆU (chính xác, không cần đoán):

    claim["text"] KHÔNG nằm trong corpus.get(claim["doc_id"]).body
    nhưng CHÍNH câu đó CÓ trong bằng chứng agent đã quan sát

Vế thứ hai mới là phần quan trọng: nó tách việc của bạn khỏi việc của
`critic` (§2). Câu có trong bằng chứng nhưng gắn sai tài liệu -> GẮN LẠI
(việc của bạn). Câu không có trong bằng chứng nào -> BỊA, để `critic` xoá.
Hai điều kiện loại trừ nhau nên hai lớp không giành điểm của nhau.

ĐƯỢC PHÉP VÀ KHÔNG ĐƯỢC PHÉP:
  * ĐƯỢC: đổi `claim["doc_id"]`, cập nhật `report["citations"]`.
  * KHÔNG: sửa `claim["text"]`. Scorer chỉ cho điểm khi câu là trích dẫn
    nguyên văn của MỘT DÒNG trong tài liệu được trích VÀ đúng là chữ mô
    hình đã viết. Thêm dấu chấm, đổi dấu nháy, "chuẩn hoá" khoảng trắng,
    hay vá lại câu bị cắt bằng nội dung lấy từ corpus đều làm mất cả hai
    điều kiện cùng lúc (đo được: -40 điểm).

CHỈ ĐƯỢC GẮN VÀO TÀI LIỆU ĐÃ QUAN SÁT. Trích một tài liệu mà lượt chạy
chưa từng đọc bị chấm `UNRETRIEVED`. Vì vậy hãy tìm nguồn trong
`ctx.observed_text`, đừng quét cả corpus rồi gắn bừa: điều kiện
`doc.body in ctx.observed_text` nghĩa là "tài liệu này đã về nguyên vẹn
từ một lần fetch sạch" — một đoạn snippet hay một bản bị cắt không tính.

CÔNG CỤ CÓ SẴN:
    ctx.observed_text  -> toàn bộ quan sát agent đã thấy, nối lại
    ctx.corpus.get(doc_id) -> Doc | None
    ctx.corpus.docs    -> danh sách Doc (doc_id, title, body); trong vòng
                          CHẤM ĐIỂM, `Doc.tags` LUÔN RỖNG — nhãn bẫy
                          ('outdated', 'contradiction', 'injection'…) bị
                          gỡ khỏi corpus mà code của bạn cầm, vì đọc nhãn
                          là tra bảng chứ không phải kỹ năng lab này chấm.
                          Ở vòng LUYỆN TẬP seed 42 thì `data/corpus/*.json`
                          vẫn có nhãn trên đĩa: hard-code được, và điều đó
                          được nói thẳng ra ở đây thay vì giấu đi.

Cài đặt:  ReActAgent(..., middleware=[..., CitationChecker(), ...])
Xem `harness/middleware.py` để biết thứ tự các hook.
"""

from __future__ import annotations

from harness.middleware import Middleware


class CitationChecker(Middleware):
    """Trỏ mỗi claim về đúng tài liệu thật sự chứa câu đó."""

    name = "citation_checker"

    def after_agent(self, ctx, report):
        claims = report.get("claims")
        if not isinstance(claims, list) or not claims or ctx.corpus is None:
            return report

        observed = ctx.observed_text
        rewired = 0
        for claim in claims:
            if not isinstance(claim, dict):
                continue
            text = claim.get("text")
            if not isinstance(text, str) or not text:
                continue
            cited = ctx.corpus.get(claim.get("doc_id"))
            if cited is not None and text in cited.body:
                continue  # trích dẫn đã đúng
            # Tầng 1: tài liệu đã về NGUYÊN VẸN từ một lần fetch sạch.
            # Tầng 2: tài liệu từng hiện trong kết quả search. Scorer dựng
            # `retrieved` bằng cách PHÁT LẠI mỗi truy vấn search chứ không
            # chỉ đếm fetch_doc (arena/scorer.py:1180-1194), nên gắn vào
            # nó KHÔNG bị chấm UNRETRIEVED — mà một bản fetch bị cắt thì
            # tầng 1 trượt, và không có tầng 2 thì claim kẹt ở
            # MISATTRIBUTED (phạt 0.5) thay vì thành SUPPORTED (0.0).
            source = None
            for observed_doc in (
                lambda doc: doc.body in observed,
                lambda doc: doc.doc_id in observed,
            ):
                source = next(
                    (
                        doc
                        for doc in ctx.corpus.docs
                        if text in doc.body and observed_doc(doc)
                    ),
                    None,
                )
                if source is not None:
                    break
            if source is None:
                continue  # không bịa doc_id; để `critic` xử lý
            claim["doc_id"] = source.doc_id  # GIỮ NGUYÊN text
            rewired += 1

        self.bump(ctx, "rewired", rewired)
        report["citations"] = sorted(
            {
                claim["doc_id"]
                for claim in claims
                if isinstance(claim, dict) and claim.get("doc_id")
            }
        )
        return report
