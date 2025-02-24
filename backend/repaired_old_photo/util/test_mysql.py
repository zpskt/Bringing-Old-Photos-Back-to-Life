if __name__ == "__main__":
    import MySQLdb
    from MySQLdb.constants import FIELD_TYPE

    # 连接到数据库
    conn = MySQLdb.connect(host="8.146.205.137", user="root", passwd="zhangpeng1211zP!", db="repair_photo")
    cursor = conn.cursor()

    # 获取列信息
    cursor.execute("DESCRIBE repair_photo")
    for row in cursor.fetchall():
        field_name, field_type = row[:2]
        if field_type == FIELD_TYPE.JSON:
            print(f"Column '{field_name}' is of type JSON")
        else:
            print(f"Column '{field_name}' is not JSON")

    # 关闭连接
    cursor.close()
    conn.close()