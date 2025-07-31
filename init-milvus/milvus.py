from pymilvus import Collection, FieldSchema, CollectionSchema, DataType, connections, utility
from sentence_transformers import SentenceTransformer
import numpy as np


def create_and_insert():
    # 0. 连接 Milvus 服务器（默认地址）
    connections.connect("default", host="localhost", port="19530")

    # 检查连接是否成功
    try:
        print(f"Milvus版本: {utility.get_server_version()}")
        print("连接成功!")
    except Exception as e:
        print(f"连接失败: {e}")

    # 1. 定义字段
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=200),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384)
    ]

    # 2. 创建 schema 和集合
    schema = CollectionSchema(fields, description="测试集合")
    collection_name = "test_collection"

    # 如果集合存在，先删除
    if utility.has_collection(collection_name):
        print(f"集合 {collection_name} 已存在，删除中...")
        utility.drop_collection(collection_name)
        print("删除完成。")

    collection = Collection(name=collection_name, schema=schema)

    # 3. 准备数据
    num_entities = 1000
    titles = [f"title_{i}" for i in range(num_entities)]

    # 使用 SentenceTransformer 提取语义向量
    # model = SentenceTransformer("all-MiniLM-L6-v2")
    model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
    embeddings = model.encode(titles, normalize_embeddings=True).tolist()

    # 4. 插入数据（不包含 auto_id 字段）
    insert_result = collection.insert(
        data=[titles, embeddings],
        fields=["title", "embedding"]
    )

    print(f"插入完成，主键ID: {insert_result.primary_keys}")

    # 5. 创建索引
    collection.create_index(
        field_name="embedding",
        index_params={
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 128}
        }
    )

    # 6. 加载集合到内存
    collection.load()

    return collection


if __name__ == "__main__":
    collection = create_and_insert()
    print("加载成功")
    # collection.release()  # 手动释放内存
