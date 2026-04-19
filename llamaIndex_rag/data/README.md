# 知识库原始文档目录

把需要构建知识库的文档放到这里，支持子目录。当前默认支持以下格式：

- PDF (`.pdf`)
- Markdown (`.md`)
- 纯文本 (`.txt`)
- Word (`.docx`)

放好文件后，在项目根目录执行：

```bash
python -m scripts.ingest
```

即可完成索引构建（结果会持久化到 `../storage/`）。
