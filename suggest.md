# Yêu cầu AI implement sync tác phẩm từ LexiLearn BE sang Mongo

Mục tiêu: khi admin CRUD tác phẩm ở BE, dữ liệu trong Postgres sẽ được sync sang AI service để AI tự chunk, tạo embedding và upsert vào Mongo.

BE là source of truth cho tác phẩm.

Mongo bên AI là index phục vụ search/RAG/chat, không phải nơi CRUD chính.

## 1. Luồng tổng thể

```text
Admin CRUD tác phẩm / section / commentary / author trên BE
        |
        v
BE cập nhật Postgres
        |
        v
Postgres transaction commit thành công
        |
        v
BE build full snapshot của work
        |
        v
BE gọi AI internal endpoint
        |
        v
AI chunk lại + tạo embedding + upsert vào Mongo
```

BE gọi sync theo kiểu best-effort:

- nếu AI sync thành công: Mongo được cập nhật;
- nếu AI service lỗi: BE chỉ log warning, không rollback CRUD Postgres.

## 2. Endpoint AI cần implement

### 2.1. Upsert/rebuild một tác phẩm

```http
PUT /internal/works/{workSlug}/sync
Content-Type: application/json
```

Endpoint này nhận full snapshot của một tác phẩm từ BE.

`workSlug` trên path phải khớp với:

```text
payload.work.slug
```

Nếu không khớp, AI nên trả `400 Bad Request`.

### 2.2. Delete/deactivate một tác phẩm

```http
DELETE /internal/works/{workSlug}/sync
```

Endpoint này được gọi khi BE xóa work.

AI nên mark inactive toàn bộ chunks theo `work_slug = workSlug`.

Khuyến nghị dùng soft delete:

```text
is_active = false
```

Không bắt buộc hard delete vì soft delete dễ debug/rollback hơn.

## 3. Payload PUT BE sẽ gửi

Schema:

```text
literature_work_snapshot.v1
```

Ví dụ payload:

```json
{
  "schema_version": "literature_work_snapshot.v1",
  "synced_at": "2026-07-19T05:00:00Z",
  "work": {
    "id": "uuid",
    "title": "TỎ LÒNG",
    "slug": "to_long",
    "original_title": null,
    "genre": "tho_ca",
    "sub_genre": "that_ngon_tu_tuyet",
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
  "author": {
    "id": "uuid",
    "name": "Phạm Ngũ Lão",
    "pen_name": null,
    "slug": "pham_ngu_lao",
    "birth_year": null,
    "death_year": null,
    "period": "trung_dai",
    "bio": "...",
    "portrait_url": null
  },
  "sections": [
    {
      "id": "uuid",
      "number": 1,
      "title": "Phiên âm",
      "content": "Hoành sóc giang sơn cáp kỷ thu...",
      "content_type": "POETRY",
      "word_count": 28
    },
    {
      "id": "uuid",
      "number": 2,
      "title": "Dịch nghĩa",
      "content": "Cầm ngang ngọn giáo trấn giữ đất nước...",
      "content_type": "PROSE",
      "word_count": 42
    }
  ],
  "commentaries": [
    {
      "id": "uuid",
      "title": "Tổng kết giá trị",
      "content": "Giá trị hiện thực: ...\n\nGiá trị nhân đạo: ...",
      "commentator_name": "LexiLearn Editorial",
      "commentator_type": "EDITORIAL",
      "source_title": null,
      "source_url": null,
      "published_year": null,
      "display_order": 0,
      "is_featured": true,
      "is_published": true
    }
  ],
  "tags": [
    {
      "id": "uuid",
      "name": "Thơ trung đại",
      "slug": "tho_trung_dai",
      "description": null
    }
  ]
}
```

Payload này là dữ liệu nguồn từ Postgres, chưa phải Mongo chunk data.

AI không nên kỳ vọng BE gửi sẵn:

- `chunk_id`;
- `embedding`;
- `token_count`;
- `search_text`;
- `model_version`;
- `position`.

Những field đó là trách nhiệm của AI khi rebuild Mongo index.

## 4. Cách xử lý khi nhận PUT

Khi nhận:

```http
PUT /internal/works/{workSlug}/sync
```

AI nên xử lý theo thứ tự:

```text
1. Validate schema_version = literature_work_snapshot.v1.
2. Validate path workSlug == payload.work.slug.
3. Mark inactive hoặc xóa toàn bộ chunk cũ theo work_slug.
4. Rebuild chunks mới từ snapshot.
5. Tạo embedding cho chunks mới.
6. Upsert chunks mới vào Mongo.
7. Set is_active = true cho chunks mới.
```

Khuyến nghị dùng transaction/batch nếu Mongo setup hỗ trợ. Nếu không, ít nhất cần đảm bảo không để tồn tại đồng thời chunk cũ active và chunk mới active cùng `work_slug`.

## 5. Data AI nên chunk từ đâu

AI nên build Mongo chunks từ các nhóm sau:

### 5.1. Nội dung đọc chính

Nguồn:

```text
payload.sections[]
```

Mapping gợi ý:

```text
chunk_category = text_section
section_slug   = slug từ section.title hoặc section.number
section_title  = section.title
section_order  = section.number
content_type   = section.content_type
content        = section.content
```

### 5.2. Tác giả

Nguồn:

```text
payload.author.bio
```

Mapping gợi ý:

```text
chunk_category = author_bio
content        = payload.author.bio
```

Nếu `author.bio = null` hoặc rỗng thì bỏ qua chunk này.

### 5.3. Bối cảnh và giá trị tác phẩm

Nguồn:

```text
payload.work.historical_context
payload.work.realistic_value
payload.work.humanistic_value
payload.work.artistic_value
payload.work.summary
payload.work.famous_quote
```

Mapping gợi ý:

```text
historical_context -> chunk_category = historical_context
realistic_value    -> chunk_category = reality_value
humanistic_value   -> chunk_category = human_value
artistic_value     -> chunk_category = art_value
summary            -> chunk_category = analysis hoặc summary
famous_quote       -> chunk_category = analysis hoặc famous_quote
```

Nếu field nào null/rỗng thì bỏ qua.

### 5.4. Bình phẩm

Nguồn:

```text
payload.commentaries[]
```

Mapping gợi ý:

```text
chunk_category = analysis
content        = commentary.title + "\n\n" + commentary.content
```

Chỉ nên chunk commentary có:

```text
is_published = true
```

Nếu muốn AI dùng cả unpublished để admin test thì cần thống nhất riêng. Mặc định nên chỉ dùng published.

### 5.5. Tags

Nguồn:

```text
payload.tags[]
```

Tags không nhất thiết tạo thành chunk riêng.

Nên đưa vào metadata/search_text để tăng khả năng tìm kiếm theo chủ đề.

## 6. Metadata Mongo cần giữ

Mỗi chunk Mongo nên giữ metadata đủ để trace ngược về BE:

```json
{
  "schema_version": "literature_seed.v1",
  "source": "be_sync",
  "work_id": "uuid",
  "work_title": "TỎ LÒNG",
  "work_slug": "to_long",
  "author_id": "uuid",
  "author_name": "Phạm Ngũ Lão",
  "author_slug": "pham_ngu_lao",
  "author_period": "trung_dai",
  "work_period": "trung_dai",
  "genre": "tho_ca",
  "sub_genre": "that_ngon_tu_tuyet",
  "grade": 10,
  "semester": 1,
  "chunk_category": "text_section",
  "section_id": "uuid",
  "section_slug": "phien_am",
  "section_title": "Phiên âm",
  "section_order": 1,
  "content_type": "POETRY"
}
```

Với chunks không thuộc section, các field section có thể null.

## 7. Quy tắc chunk_id để tránh duplicate

AI nên generate `chunk_id` deterministic theo work + source + index.

Ví dụ:

```text
to_long__text_section__phien_am__001
to_long__text_section__dich_nghia__001
to_long__author_bio__001
to_long__historical_context__001
to_long__reality_value__001
to_long__human_value__001
to_long__art_value__001
to_long__commentary__000__001
```

Không nên dùng UUID random cho `chunk_id` nếu mỗi lần sync lại sẽ tạo document mới và khó dọn duplicate.

Khuyến nghị unique key/index trong Mongo:

```text
work_slug + chunk_id
```

Hoặc:

```text
source_doc_id + chunk_id
```

Trong đó `source_doc_id` có thể là:

```text
work:{work_slug}
```

## 8. Delete behavior

Khi nhận:

```http
DELETE /internal/works/{workSlug}/sync
```

AI nên:

```text
updateMany({ "metadata.work_slug": workSlug }, { "$set": { "is_active": false } })
```

hoặc filter tương đương theo field đang dùng.

Không cần gọi embedding.

Không cần rebuild.

## 9. Response format đề xuất

PUT success:

```json
{
  "success": true,
  "work_slug": "to_long",
  "chunks_upserted": 12,
  "chunks_deactivated": 12
}
```

DELETE success:

```json
{
  "success": true,
  "work_slug": "to_long",
  "chunks_deactivated": 12
}
```

Nếu lỗi validate:

```json
{
  "success": false,
  "detail": "path workSlug does not match payload.work.slug"
}
```

## 10. Lưu ý về author-only CRUD

BE hiện sync theo work, không sync author độc lập.

Trường hợp:

```text
Admin tạo author mới nhưng author chưa có work
```

BE không gọi AI, vì chưa có nội dung tác phẩm để chunk/index.

Trường hợp:

```text
Admin sửa author đã có work
```

BE sẽ tìm tất cả work thuộc author đó và gọi:

```http
PUT /internal/works/{workSlug}/sync
```

cho từng work.

Vì snapshot work có kèm object `author`, Mongo sẽ được cập nhật thông tin tác giả thông qua rebuild chunks của từng tác phẩm.

Nếu AI muốn quản lý author độc lập trong Mongo thì cần thêm contract riêng:

```http
PUT /internal/authors/{authorSlug}/sync
DELETE /internal/authors/{authorSlug}/sync
```

Nhưng với RAG hiện tại theo tác phẩm, chưa cần làm author-only sync.

## 11. BE endpoint/config hiện tại

BE sẽ gọi AI bằng:

```text
AI_BASE_URL
app.ai.base-url
```

Ví dụ khi BE chạy trong Docker và AI chạy ngoài host:

```text
AI_BASE_URL=http://host.docker.internal:8000
```

Nếu AI endpoint chưa implement, CRUD BE vẫn thành công nhưng log sẽ có warning.

## 12. Checklist bên AI cần làm

- Tạo route `PUT /internal/works/{workSlug}/sync`.
- Tạo route `DELETE /internal/works/{workSlug}/sync`.
- Validate `schema_version`.
- Validate path `workSlug` khớp `payload.work.slug`.
- Implement deactivate old chunks by `work_slug`.
- Implement rebuild chunks from sections/author/work values/commentaries.
- Generate deterministic `chunk_id`.
- Generate embedding.
- Upsert Mongo.
- Chỉ query chunks có `is_active = true` khi RAG/search.
- Trả response JSON có số chunk upsert/deactivate để BE log/debug dễ.
