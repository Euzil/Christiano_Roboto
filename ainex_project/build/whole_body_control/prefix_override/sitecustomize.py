import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/workspaces/workspaces/Christiano_Roboto/ainex_project/install/whole_body_control'
