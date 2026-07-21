# BE -> AI sync contract mới

Mục tiêu thay đổi: tách dữ liệu tác giả ra khỏi work sync để tránh lặp full author data trong nhiều tác phẩm cùng tác giả.

Lưu ý quan trọng: format data AI gửi về BE để import seed data vẫn giữ nguyên như hiện tại trong `docs/data/data*.json`. Thay đổi dưới đây chỉ áp dụng cho luồng BE sync realtime sang AI khi admin CRUD.

## 1. Work sync

Endpoint AI cần support:

```http
PUT /internal/works/{workSlug}/sync
DELETE /internal/works/{workSlug}/sync
```

Khi BE gọi `PUT`, payload mới sẽ không còn field `author` full nữa. Thay bằng `author_ref` nhẹ:

```json
{
  "schema_version": "literature_work_snapshot.v1",
  "synced_at": "2026-07-21T05:00:00Z",
  "work": {
    "id": "uuid",
    "title": "CẢNH NGÀY HÈ",
    "slug": "canh_ngay_he",
    "original_title": null,
    "genre": "tho_ca",
    "sub_genre": "tho_nom_duong_luat",
    "period": "trung_dai",
    "grade": 10,
    "semester": 1,
    "publish_year": null,
    "summary": "...",
    "cover_url": null,
    "is_published": true,
    "historical_context": "...",
    "realistic_value": "...",
    "humanistic_value": "...",
    "artistic_value": "...",
    "famous_quote": null,
    "quote_attribution": null
  },
  "author_ref": {
    "id": "uuid",
    "slug": "nguyen_trai"
  },
  "sections": [
    {
      "id": "uuid",
      "number": 1,
      "title": "Nội dung bài thơ",
      "content": "Rồi hóng mát thuở ngày trường...",
      "content_type": "POETRY",
      "word_count": 54
    }
  ],
  "commentaries": [],
  "tags": []
}
```

AI xử lý:

- Dùng `work.slug` làm khóa chính để rebuild chunks của tác phẩm.
- `sections`, `commentaries`, `tags` luôn là array, nếu chưa có data thì là `[]`, không phải `null`.
- `author_ref` chỉ dùng để liên kết sang author document riêng.
- Không lấy `bio`, `portrait_url`, `birth_year`, `death_year` từ work payload nữa.

Khi BE gọi `DELETE /internal/works/{workSlug}/sync`, AI mark inactive hoặc xóa toàn bộ chunks/index theo `work_slug`.

## 2. Author sync

Endpoint AI cần support thêm:

```http
PUT /internal/authors/{authorSlug}/sync
DELETE /internal/authors/{authorSlug}/sync
```

Khi BE gọi `PUT`, payload:

```json
{
  "schema_version": "literature_author_snapshot.v1",
  "synced_at": "2026-07-21T05:00:00Z",
  "author": {
    "id": "uuid",
    "name": "Nguyễn Trãi",
    "pen_name": "Ức Trai",
    "slug": "nguyen_trai",
    "birth_year": 1380,
    "death_year": 1442,
    "period": "trung_dai",
    "bio": "Nguyễn Trãi là nhà chính trị, nhà quân sự, nhà văn hóa lớn...",
    "portrait_url": "https://..."
  }
}
```

AI xử lý:

- Upsert author document theo `author.slug` hoặc `author.id`.
- Nếu cần semantic search tiểu sử tác giả thì chunk/embed riêng từ `author.bio`.
- Các work chunks chỉ reference bằng `author_ref.id` và `author_ref.slug`.

Khi BE gọi `DELETE /internal/authors/{authorSlug}/sync`, AI mark inactive hoặc xóa author document/index theo `author_slug`.

## 3. Khi nào BE gọi sync

Work sync:

- Admin create work -> `PUT /internal/works/{workSlug}/sync`
- Admin update work -> `PUT /internal/works/{workSlug}/sync`
- Admin delete work cover -> `PUT /internal/works/{workSlug}/sync`
- Admin delete work -> `DELETE /internal/works/{workSlug}/sync`
- Admin create/update/delete section -> `PUT /internal/works/{workSlug}/sync`
- Admin create/update/delete commentary -> `PUT /internal/works/{workSlug}/sync`

Author sync:

- Admin create author -> `PUT /internal/authors/{authorSlug}/sync`
- Admin update author -> `PUT /internal/authors/{authorSlug}/sync`
- Admin delete author portrait -> `PUT /internal/authors/{authorSlug}/sync`
- Admin delete author -> `DELETE /internal/authors/{authorSlug}/sync`

Ghi chú: BE hiện không cho xóa author nếu còn work liên kết, nên author delete chỉ xảy ra với author không còn tác phẩm.

## 4. Những gì BE đã sửa

- Work payload đổi từ full `author` sang `author_ref`.
- Thêm `AuthorSyncPayload` riêng với schema `literature_author_snapshot.v1`.
- Thêm async event riêng cho author sync, chạy sau DB commit giống work sync.
- Thêm client call:
  - `PUT /internal/authors/{authorSlug}/sync`
  - `DELETE /internal/authors/{authorSlug}/sync`
- CRUD author không còn sync lại toàn bộ works của author nữa; chỉ sync author snapshot riêng.

## 5. Checklist cho AI

- Support đủ 4 endpoint:
  - `PUT /internal/works/{workSlug}/sync`
  - `DELETE /internal/works/{workSlug}/sync`
  - `PUT /internal/authors/{authorSlug}/sync`
  - `DELETE /internal/authors/{authorSlug}/sync`
- Work sync không expect field `author` full nữa.
- Work sync đọc `author_ref.id` và `author_ref.slug`.
- Author sync lưu/index dữ liệu tác giả riêng.
- PUT nhiều lần cùng một work/author không tạo duplicate active documents.
- DELETE dùng soft delete `is_active=false` hoặc xóa cứng tùy AI, nhưng không để active stale data.
