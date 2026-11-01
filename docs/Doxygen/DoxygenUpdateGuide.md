##Steps to update Doxygen documentation

1) Delete old Doxygen documentation, located in `/docs/Doxygen/html/`
```
rm -rf /docs/Doxygen/html
```

2) Run Doxygen to generate new documentation
```
doxygen docs/Doxygen/Doxyfile
```
3) Push changes to github