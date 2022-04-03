data <-read.csv("C:/Berkeley/CS282A/GradProject/CS282_final_project/DATA/carbon_nanotubes_reformat.csv")

dt = sort(sample(nrow(data), nrow(data)*.7))
train<-data[dt,]
test<-data[-dt,]

write.csv(train,"C:/Berkeley/CS282A/GradProject/CS282_final_project/DATA/carbonnanotubes_train.csv", row.names = FALSE)
write.csv(test,"C:/Berkeley/CS282A/GradProject/CS282_final_project/DATA/carbonnanotubes_test.csv", row.names = FALSE)
