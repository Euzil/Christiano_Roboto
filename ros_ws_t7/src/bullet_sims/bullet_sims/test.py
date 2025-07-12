#!/usr/bin/env python3
"""
简化的 Talos 测试 - 专注于检查必要的函数是否存在
"""

import sys
sys.path.append('.')

def test_tsid_wrapper_functions():
    """测试 TSIDWrapper 是否有必要的 walking 函数"""
    
    print("\n" + "="*60)
    print("测试: TSIDWrapper Walking 函数检查")
    print("="*60)
    
    try:
        from bullet_sims.tsid_wrapper import TSIDWrapper
        import bullet_sims.talos_conf as conf
        
        print("1. 创建 TSIDWrapper...")
        stack = TSIDWrapper(conf)
        print("✓ TSIDWrapper 创建成功")
        
        print("\n2. 检查必需的 walking 函数是否存在:")
        
        required_functions = [
            'activateContact',
            'deactivateContact', 
            'activateTask',
            'deactivateTask',
            'setTaskReference',
            'setComReference',
            'getFramePose',
            'getCenterOfMass',
            'getCenterOfMassVelocity',
            'getContactForces',
            'solve',
            '_setup_foot_contacts_and_tasks',
            '_setup_walking_tasks'
        ]
        
        missing_functions = []
        existing_functions = []
        
        for func_name in required_functions:
            if hasattr(stack, func_name):
                existing_functions.append(func_name)
                print(f"   ✓ {func_name}")
            else:
                missing_functions.append(func_name)
                print(f"   ✗ {func_name} - 缺失")
        
        print(f"\n结果: {len(existing_functions)}/{len(required_functions)} 函数存在")
        
        if missing_functions:
            print(f"\n❌ 缺少以下函数:")
            for func in missing_functions:
                print(f"   - {func}")
            
            print(f"\n📝 你需要在 TSIDWrapper 类中添加这些函数:")
            print("   请参考之前的对话，添加这些 walking 控制所需的函数")
            return False
        else:
            print(f"\n✅ 所有必需的函数都存在!")
            return True
            
    except Exception as e:
        print(f"✗ TSIDWrapper 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tsid_wrapper_attributes():
    """测试 TSIDWrapper 是否有必要的属性"""
    
    print("\n" + "="*60)
    print("测试: TSIDWrapper Walking 属性检查")
    print("="*60)
    
    try:
        from bullet_sims.tsid_wrapper import TSIDWrapper
        import bullet_sims.talos_conf as conf
        
        stack = TSIDWrapper(conf)
        
        print("检查必需的属性:")
        
        required_attributes = [
            'active_contacts',
            'active_tasks',
            'contact_tasks', 
            'motion_tasks'
        ]
        
        missing_attributes = []
        existing_attributes = []
        
        for attr_name in required_attributes:
            if hasattr(stack, attr_name):
                existing_attributes.append(attr_name)
                print(f"   ✓ {attr_name}")
            else:
                missing_attributes.append(attr_name)
                print(f"   ✗ {attr_name} - 缺失")
        
        if missing_attributes:
            print(f"\n❌ 缺少以下属性:")
            for attr in missing_attributes:
                print(f"   - {attr}")
            
            print(f"\n📝 你需要在 TSIDWrapper.__init__ 中添加:")
            print("   self.active_contacts = {}")
            print("   self.active_tasks = {}")
            print("   self.contact_tasks = {}")
            print("   self.motion_tasks = {}")
            return False
        else:
            print(f"\n✅ 所有必需的属性都存在!")
            return True
            
    except Exception as e:
        print(f"✗ 属性测试失败: {e}")
        return False

def test_basic_functionality():
    """测试基本功能是否工作"""
    
    print("\n" + "="*60)
    print("测试: 基本功能测试")
    print("="*60)
    
    try:
        from bullet_sims.tsid_wrapper import TSIDWrapper
        import bullet_sims.talos_conf as conf
        
        stack = TSIDWrapper(conf)
        
        print("1. 测试基本状态查询...")
        
        # 测试质心
        try:
            com_pos = stack.getCenterOfMass()
            print(f"   ✓ 质心位置: [{com_pos[0]:.3f}, {com_pos[1]:.3f}, {com_pos[2]:.3f}]")
        except Exception as e:
            print(f"   ✗ 质心位置获取失败: {e}")
            return False
        
        # 测试质心速度
        try:
            com_vel = stack.getCenterOfMassVelocity()
            print(f"   ✓ 质心速度: [{com_vel[0]:.3f}, {com_vel[1]:.3f}, {com_vel[2]:.3f}]")
        except Exception as e:
            print(f"   ✗ 质心速度获取失败: {e}")
            return False
        
        # 测试 frame 位姿
        try:
            left_pose = stack.getFramePose(conf.lf_frame_name)
            print(f"   ✓ 左脚位置: [{left_pose[0,3]:.3f}, {left_pose[1,3]:.3f}, {left_pose[2,3]:.3f}]")
        except Exception as e:
            print(f"   ✗ 左脚位置获取失败: {e}")
            return False
        
        # 测试接触力
        try:
            forces = stack.getContactForces()
            print(f"   ✓ 接触力获取成功: {list(forces.keys())}")
        except Exception as e:
            print(f"   ✗ 接触力获取失败: {e}")
            return False
        
        print("\n✅ 基本功能测试通过!")
        return True
        
    except Exception as e:
        print(f"✗ 基本功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_next_steps():
    """显示下一步的操作指南"""
    
    print("\n" + "="*80)
    print("下一步操作指南")
    print("="*80)
    
    print("\n如果测试失败，请按照以下步骤:")
    print("\n1. 添加必需的属性到 TSIDWrapper.__init__:")
    print("   在 TSIDWrapper.__init__ 的末尾添加:")
    print("   ```python")
    print("   # 跟踪激活的接触和任务")
    print("   self.active_contacts = {}")
    print("   self.active_tasks = {}")
    print("   self.contact_tasks = {}")
    print("   self.motion_tasks = {}")
    print("   ```")
    
    print("\n2. 添加必需的函数到 TSIDWrapper 类:")
    print("   参考之前对话中提供的函数实现")
    print("   重点函数包括:")
    print("   - activateContact() / deactivateContact()")
    print("   - activateTask() / deactivateTask()")
    print("   - setTaskReference() / setComReference()")
    print("   - getFramePose() / getCenterOfMass() / getCenterOfMassVelocity()")
    print("   - getContactForces()")
    print("   - solve()")
    
    print("\n3. 修复 Talos.py 中的方法调用:")
    print("   将 self.sim.time() 改为 self.sim.simTime()")
    print("   将 self.sim.dt() 改为固定值或从配置获取")
    
    print("\n4. 重新运行这个测试确认修复")

def main():
    """主测试函数"""
    
    print("TSIDWrapper Walking 功能检查测试")
    print("这个测试会检查你是否已经添加了必要的 walking 控制函数")
    
    # 运行测试
    tests = [
        ("TSIDWrapper 函数检查", test_tsid_wrapper_functions),
        ("TSIDWrapper 属性检查", test_tsid_wrapper_attributes),
        ("基本功能测试", test_basic_functionality),
    ]
    
    results = []
    all_passed = True
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
            if not result:
                all_passed = False
        except Exception as e:
            print(f"❌ {test_name} 发生异常: {e}")
            results.append((test_name, False))
            all_passed = False
    
    # 显示结果总结
    print("\n" + "="*80)
    print("测试结果总结")
    print("="*80)
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name:25} : {status}")
    
    if all_passed:
        print("\n🎉 所有测试都通过了!")
        print("你现在可以:")
        print("1. 修复 Talos.py 中的 time() 方法调用")
        print("2. 开始实现完整的 walking 控制")
    else:
        print(f"\n⚠️ 有测试失败")
        show_next_steps()

if __name__ == "__main__":
    main()