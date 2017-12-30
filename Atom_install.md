# Atom_install on CentOs 6.8

1. Download the atom file from https://atom.io/

2. Install Atom

3. Deal with Dependenes
```

./configure --prefix=/home/hughli/Templates/log/

configure: error: Your intltool is too old.  You need intltool 0.35.0 or later.


swtich to root

yum list intltool

yum install intltool

./configure --prefix=/home/hughli/Templates/log/

configure: error: Package requirements (glib-2.0 >= 2.38.0
	gio-2.0
	gio-unix-2.0) were not met:

No package 'glib-2.0' found
No package 'gio-2.0' found
No package 'gio-unix-2.0' found
```
