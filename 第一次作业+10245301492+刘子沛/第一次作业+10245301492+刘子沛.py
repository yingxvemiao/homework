stu={}

def menu():
    print("-----------宿舍管理系统-----------")
    print("1. 按学号查找某一位学生的具体信息")
    print("2. 录入新的学生信息")
    print("3. 显示现有的所有学生信息")
    print("4. 退出系统")
    print("---------------------------------")

def query():
    id=input("请输入要查找的学生学号: ").strip()
    
    if not id:
        print("错误：学号不能为空！")
        return
    
    if id in stu:
        info=stu[id]
        print("找到学生信息:")
        print(f"    学号: {id}")
        print(f"    姓名: {info['name']}")
        print(f"    性别: {info['gender']}")
        print(f"    宿舍房间号: {info['room']}")
        print(f"    联系电话: {info['phone']}")
    else:
        print(f"未找到学号为 {id} 的学生信息")

def insert():
    while True:
        id=input("请输入学号: ").strip()
        if not id:
            print("错误：学号不能为空！")
            continue
        if not id.isdigit() or len(id) != 11:
            print("错误：学号必须为11位数字！")
            continue
        if id in stu:
            print("错误：该学号已存在！")
            continue
        break
    
    while True:
        name=input("请输入姓名: ").strip()
        if not name:
            print("错误：姓名不能为空！")
            continue
        break
    
    while True:
        gender=input("请输入性别(男/女): ").strip()
        if gender not in ['男', '女']:
            print("错误：性别只能是'男'或'女'！")
            continue
        break
    
    while True:
        room=input("请输入宿舍房间号: ").strip()
        if not room:
            print("错误：宿舍房间号不能为空！")
            continue
        if not room.isdigit() or len(room) != 3:
            print("错误：房间号必须为3位数字！")
            continue
        break
    
    while True:
        phone=input("请输入联系电话: ").strip()
        if not phone:
            print("错误：联系电话不能为空！")
            continue
        if not phone.isdigit() or len(phone) != 11:
            print("错误：电话必须为11位数字！")
            continue
        break
    
    confirm=input("确认添加?(y/n): ")
    if confirm.lower()=='y':
        stu[id]={'name':name,'gender':gender,'room':room,'phone':phone}
        print("已录入新的学生信息！")
    else:
        print("未录入新的学生信息！")

def show_all():
    if not stu:
        print("当前没有学生信息")
        return
    
    print("-----------所有学生信息-----------")
    for id,info in stu.items():
        print(f"学号: {id}")
        print(f"姓名: {info['name']}")
        print(f"性别: {info['gender']}")
        print(f"宿舍房间号: {info['room']}")
        print(f"联系电话: {info['phone']}")
        print("---------------------------------")

print("欢迎使用宿舍管理系统！")
while True:
    menu()
    choice=input("请选择操作(1-4): ").strip()
    if choice=='1':
        query()
    elif choice=='2':
        insert()
    elif choice=='3':
        show_all()
    elif choice=='4':
        print("感谢使用宿舍管理系统，再见！")
        break
    else:
        print("错误：请输入1-4之间的数字！")
    input("按回车键继续......")
