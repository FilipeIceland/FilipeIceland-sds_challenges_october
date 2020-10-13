library(shiny)
library(dplyr)
library(ggplot2)
library(ggcorrplot)
library(corrplot)
library(rcompanion)
library(randomForest)
library(caret)
library(rpart)
library(shinyjs)
library(leaps)
require(stats)
library(FSelectorRcpp)
library(OneR)
require(MASS)
library(reshape)
library(fastDummies)
library(Metrics)

df <- read.csv("data/public_cars.csv")
df_pred <- read.csv("data/pred_cars.csv")
all_names <- names(df)
target_name <- 'price_usd'
var_names <- all_names[all_names != target_name]
num_names <- var_names[unlist(lapply(df[,var_names], is.numeric))]
n_num <- length(num_names)
n_num2 <- n_num +1
cat_names <- var_names[!(var_names %in% num_names)]
n_cat <- length(cat_names)
all_num_names <- c(num_names,target_name)
df_all_num <- df[,all_num_names]
corr_num <- cor(df_all_num, use = "complete.obs")

show_notification <- function(message, duration = NULL){
  return(tryCatch(showNotification(message, duration = duration), error = function(e){}, warning = function(e){}, finally ={}))
} 
get_col_names <- function(df, type = "all", target_name = 'price_usd'){
  all_names <- names(df)
  if(type == "all"){
    return(all_names)
  }
  var_names <- all_names[all_names != target_name]
  if(type == "variables"){
    return(var_names)
  }
  num_names <- var_names[unlist(lapply(df[,var_names], is.numeric))]
  if(type == "num"){
    return(num_names)
  }
  if(type == "all_num"){
    return(c(num_names, target_name))
  }
  cat_names <- var_names[!(var_names %in% num_names)]
  if(type == "cat"){
    return(cat_names) 
  }
  bool_names <- cat_names[unlist(lapply(df[,cat_names], is.logical))]
  if(type == 'bool'){
    return(bool_names)
  }
  return(c())
}


run_cramersV <- function(newdf){
  cat_names <- get_col_names(newdf, "cat")
  n_cat <- length(cat_names)
  n_comb <- n_cat*n_cat
  cramers <- data.frame(matrix(rep(1,n_comb), ncol = n_cat))
  rownames(cramers) <- cat_names
  colnames(cramers) <- cat_names
  c <- 1
  for (i in 1:(n_cat-1)){
    cat.name <- cat_names[i]
    showNotification(cat.name, duration = 15)
    for (j in (i+1):n_cat){
      cat2.name <- cat_names[j]
      cramers[j,i] <- cramerV(newdf[,cat.name], newdf[,cat2.name])
    }
  }
  return(cramers)
}




missing_values <- function(newdf, type, target_name = "price_usd"){
  col_names <- get_col_names(newdf, type)
  missing_v <- cbind(column = col_names, missing = sapply(newdf[,col_names], function(x) sum(is.na(x))))
  return(missing_v)
}

create_new_factor <- function(traindf, cat_names = get_col_names(traindf, 'cat', target), new_factor = 'created_category', target ='price_usd', testdf = data.frame()){
  for(c in cat_names){
    levels(traindf[, c]) <- c(levels(traindf[, c]), new_factor)
    if(c%in% names(testdf)){
      levels(testdf[, c]) <- c(levels(testdf[, c]), new_factor)
    }
  }
  return(list(train = traindf, test = testdf))
}

replace_factors <- function(traindf, col, cond_col, old_values, new_value){
  levels(traindf[[col]]) <- c(levels(traindf[[col]]), new_value)
  traindf[traindf[,cond_col] %in% old_values, col] <- new_value
  return(traindf)
}


CV_lm <- function(newdf, target_name = 'price_usd'){
  
  d_newdf <- dummy_columns(newdf, remove_first_dummy = FALSE, remove_selected_columns = TRUE)
  d_newdf <- adjust_names_df(d_newdf, target_name)
  
  N <- nrow(newdf)
  
  x_sample <- sample(1:N, N)
  x_samples <- list(x_sample[1:floor(N/3)], x_sample[(floor(N/3)+1):floor(2*N/3)], x_sample[(floor(2*N/3)+1):N])
  
  
  sets_train <- list(c(x_samples[[1]],x_samples[[2]]), c(x_samples[[1]],x_samples[[3]]), c(x_samples[[2]],x_samples[[3]]))
  sets_test <- list(x_samples[[3]], x_samples[[2]], x_samples[[1]])
  
  
  trees <- list()
  rmses <- c()
  
  for(it in 1:3){
    tempdf <- d_newdf
    
    traindf <- d_newdf[sets_train[[it]],]
    testdf <- d_newdf[sets_test[[it]],]
    N2 <- nrow(traindf)
    x_sample2 <- sample(1:N2, N2)
    x_samples2 <- list(x_sample[1:floor(N2/3)], x_sample[(floor(N2/3)+1):floor(2*N2/3)], x_sample[(floor(2*N2/3)+1):N2])

    cc <- show_notification('running decision trees', duration = NULL)
    
    final_tree1 <- rpart(as.formula(paste0(target_name,' ~.')), data = traindf[x_samples2[[1]],])
    final_tree2 <- rpart(as.formula(paste0(target_name,' ~.')), data = traindf[x_samples2[[2]],])
    final_tree3 <- rpart(as.formula(paste0(target_name,' ~.')), data = traindf[x_samples2[[3]],])
    
    tempdf$tree1 <- predict(final_tree1, d_newdf)
    tempdf$tree2 <- predict(final_tree2, d_newdf)
    tempdf$tree3 <- predict(final_tree3, d_newdf)
    
    
    cc <- show_notification('running linear regression', duration = NULL)
    final_lm <- glm(as.formula(paste0(target_name,'~.')), tempdf[sets_train[[it]],], family = Gamma(link =  "log"))
    preds <- exp(predict(final_lm, tempdf[sets_test[[it]],]))
    
    new_rmse <- rmse(tempdf[sets_test[[it]],target_name], preds)
    rmses <- c(rmses, new_rmse)
    }
  return(rmses)
}
  

forward_sel <- function(newdf, nv = 10, target = 'price_usd'){
  set.seed(1234)
  var_names <- get_col_names(newdf, type = "variables", target_name = target)
  vars_selected <- c()
  lowest_RMSE <- 1000*sum(newdf[[target]])
  next_variable <- c()
  for (k in 1:nv){
    temp_RMSE <- lowest_RMSE
    for(v in var_names){
      if(!(v %in% vars_selected)){
        temp_vars <- c(vars_selected, v)
        lmFit <- train(as.formula(paste0(target, "~.")), 
                       data = newdf[,c(temp_vars,target)], 
                       method = "lm", metric = "RMSE", 
                       trControl = trainControl(method = "cv", number = 3))
        if(lmFit$results$RMSE<temp_RMSE){
          temp_RMSE <- lmFit$results$RMSE
          next_variable <- v
        }
      }
    }
    cc <- show_notification(paste0(toString(k), ' - RMSE: ', toString(round(temp_RMSE))), duration = NULL)
    if(temp_RMSE < lowest_RMSE){
      lowest_RMSE <- temp_RMSE
      vars_selected <- c(vars_selected, next_variable)
    }
    else{
      break
    }
    
  }
  return(list(vars = vars_selected, RMSE = lowest_RMSE))
}

fill_nas <- function(traindf,target = 'price_usd', testdf = data.frame()){
  var_names <- get_col_names(traindf, type = "variables", target_name = target)
  col_with_nas <- var_names[sapply(traindf[,var_names], function(x) sum(is.na(x))>0)]
  df_complete_cases <- traindf[complete.cases(traindf), var_names]
  col_with_nas2 <- c()
  col_with_nas_p <- c()
  if(nrow(testdf)>0){
    var_names2 <- get_col_names(testdf, type = "variables")
    col_with_nas_p <- var_names2[sapply(testdf[, var_names2], function(x) sum(is.na(x))>0)]
    col_with_nas2 <- col_with_nas_p[col_with_nas_p %in% var_names]
  }
  col_with_nas_f <- unique(c(col_with_nas, col_with_nas2))
  for(na_col in col_with_nas_f){
    to_fill <- !complete.cases(traindf[, na_col])
    tree <- rpart( eval(as.name(na_col)) ~ ., data = df_complete_cases)
    if(na_col %in% col_with_nas){
      traindf[to_fill, na_col] <- predict(tree, traindf[to_fill,])
    }
    if(na_col %in% col_with_nas_p){
      to_fill_p <- !complete.cases(testdf[, na_col])
      testdf[to_fill_p, na_col] <- predict(tree, testdf[to_fill_p,])
    }    
  }
  return(list(train = traindf, test = testdf))
}

bin_entropy <- function(traindf, target = 'price_usd', nbins = 3, testdf = data.frame()){
  num_names <- get_col_names(traindf, 'num', target)
  target_values <- traindf[,target]
  traindf[,target_name] <- bin(traindf[,target], nbins)
  traindf <- discretize(as.formula(paste0(target,' ~.')), traindf)
  traindf[,target_name] <- target_values
  n_p <- nrow(testdf)
  if(n_p>0){
    for (n_name in num_names){
      uniq <- levels(unique(traindf[, n_name]))
      low <- as.numeric( sub('\\((.+),.*', '\\1', uniq))[1]
      upp <- as.numeric( sub('[^,]*,([^]]*)\\]', '\\1', uniq))
      brks <- c(low, upp)
      testdf[, n_name] <- discretize(data.frame(x = testdf[,n_name]), data.frame(y = testdf[,n_name]),
                                     control = customBreaksControl(breaks = brks))$x
      #testdf[, n_name] <- ordered(testdf[, n_name], levels = levels(traindf[, n_name]))
    }    
  }
  return(list(train = traindf, test = testdf))
}

adjust_names_df <- function(newdf, target_name = 'price_usd'){
  d_n_cols <- ncol(newdf)
  
  d_varnames <- (function(ee){
    for(ii in 1:d_n_cols){
      if(ee[ii]==TRUE){
        ee[ii]<-paste0("V",ii)
      }
      else{
        ee[ii]<-target_name
      }
    }
    return(ee)
  })(!(names(newdf)%in%target_name))
  
  names(newdf) <- d_varnames
  return(newdf)
}

bool_to_binary_df <-  function(newdf, target_name = 'price_usd'){
  bool_names <- get_col_names(newdf,'bool')
  for(b in bool_names){
    newdf[, b] <- as.integer(newdf[, b])
  }
  return(newdf)
}
Rscript <- "#manipulate df_final

df_final <- fill_nas(df)$train

df_final$model_name <- paste(df_final$manufacturer_name, df_final$model_name)
man_model_group <- df_final %>% group_by(model_name) %>% summarize(n = n())
relevant_models <- man_model_group$model_name[man_model_group$n >= 10]
set_to_other <- df_final$model_name[!(df_final$model_name %in%  relevant_models)]
df_final <- replace_factors(df_final, 'model_name', 'model_name', set_to_other, 'created_category')
df_final <- replace_factors(df_final, 'engine_fuel', 'engine_fuel', c('electric','hybrid-petrol','hybrid-diesel'), 'created_category')
#df_final <- bin_entropy(df_final, target = 'price_usd', nbins = 3)$train




"
Run_model <- function(df, df_pred){
  cc <- show_notification('pre-processing', duration = NULL)
  df_final <- df
  df_pred_final <- df_pred
  n_data <- nrow(df)
  n_pred <- nrow(df_pred)
  df_pred_final$price_usd <- replicate(n_pred, 0)
  
  data_rows <- 1:n_data
  pred_rows <- (n_data+1):(n_data+n_pred)
  
  df_all <- data.frame(rbind(df_final, df_pred_final))
  #df_all <- bool_to_binary_df(df_all)
  df_all <- replace_factors(df_all, 'engine_fuel', 'engine_fuel', c('electric','hybrid-petrol','hybrid-diesel'), 'created_category')
  df_all <- fill_nas(df_all, target = 'price_usd')$train
  
  df_all$model_name <- paste(df_all$manufacturer_name, df_all$model_name)
  man_model_group <- df_all[data_rows,] %>% group_by(model_name) %>% summarize(n = n())
  relevant_models <- man_model_group$model_name[man_model_group$n >= 10]
  set_to_other <- df_all$model_name[!(df_all$model_name %in%  relevant_models)]
  df_all <- replace_factors(df_all, 'model_name', 'model_name', set_to_other, 'created_category')
  
  #df_binned <- bin_entropy(df_all[data_rows,], target = 'price_usd', nbins = 3, df_all[pred_rows,])
  #df_all <- data.frame(rbind(df_binned$train, df_binned$test ))
  
  d_df_all <- dummy_columns(df_all, remove_first_dummy = FALSE, remove_selected_columns = TRUE)
  d_df_all <- adjust_names_df(d_df_all, target_name)
  x_sample <- sample( data_rows, n_data)
  x_samples <- list(x_sample[1:floor(n_data/3)], x_sample[(floor(n_data/3)+1):floor(2*n_data/3)], x_sample[(floor(2*n_data/3)+1):n_data])
  cc <- show_notification('variable importance with decision tree', duration = NULL)
  

  #DT <- rpart(as.formula(paste0(target_name,' ~.')), data = d_df_all)
  #namesDT <- c(names(DT$variable.importance), target_name)
  namesDT <-names(d_df_all)
  
  cc <- show_notification('running decision trees', duration = NULL)
  final_tree1 <- rpart(as.formula(paste0('log(',target_name,') ~.')), data = d_df_all[x_samples[[1]],namesDT])
  final_tree2 <- rpart(as.formula(paste0('log(',target_name,') ~.')), data = d_df_all[x_samples[[2]],namesDT])
  final_tree3 <- rpart(as.formula(paste0('log(',target_name,') ~.')), data = d_df_all[x_samples[[3]],namesDT])
  d_df_all$tree1 <- predict(final_tree1, d_df_all)
  d_df_all$tree2 <- predict(final_tree2, d_df_all)
  d_df_all$tree3 <- predict(final_tree3, d_df_all)


  cc <- show_notification('running final linear regression', duration = NULL)
  
  train_df <- d_df_all[data_rows,]
  new_data_df <- droplevels.data.frame(d_df_all[pred_rows,])
  final_lm <- glm(as.formula(paste0(target_name,'~.')), train_df, family = Gamma(link =  "log"))
  
  df_pred_final$price_usd <- exp(predict(final_lm,  new_data_df))
  write.csv(df_pred_final, 'output.csv', row.names = FALSE)
  
  cc <- show_notification('plot density')
  
  predictions <- df_pred_final
  melted_data <- melt(list(prices = df$price_usd, predictions = predictions$price_usd))
  ggplot(data = melted_data, aes(value, fill=L1)) +  geom_density()
  ggsave("density.png")
  
  return(df_pred_final)
}



df_final <- df


ui <- fluidPage(

    titlePanel("Car challenge"),
    
    navbarPage("My Application",
        tabPanel("Original data",
            tabsetPanel(
              tabPanel("Categorical variables",
                  sidebarLayout(
                      sidebarPanel(                        
                          selectInput(
                                  inputId = "selectCatID",
                                  label = "Select categorical variable",
                                  choices = cat_names
                                  ),
                          selectInput(
                            inputId = "selectTargetID",
                            label = "Select numerical variable",
                            choices = all_num_names,
                            selected = target_name
                          ),
                          tableOutput(outputId = 'missingCatID')
                          ),
                      mainPanel(
                          plotOutput(outputId = "catbarPlot"),
                          plotOutput(outputId = "catboxPlot")
                          )
                      )
                  ),  
              tabPanel("Numerical variables",
                       sidebarLayout(
                           sidebarPanel(
                               selectInput(
                                   inputId = "selectNumID",
                                   label = "Select numerical variable",
                                   choices = all_num_names
                               ),
                                 selectInput(
                                   inputId = "selectNum2ID",
                                   label = "Select target",
                                   choices = all_num_names,
                                   selected = target_name
                                 ),
                               tableOutput(outputId = 'missingNumID'),
                               plotOutput(outputId = "corrPlot")
                               
                           ),
                           mainPanel(
                               plotOutput(outputId = "numlinePlot"),
                               plotOutput(outputId = "numPlot")
                           )
                      )        
              ),
              tabPanel("Calculate other metrics",
                         actionButton(inputId = "FbuttonID",
                                      label = "Calculate F-Statistic"),
                       actionButton(inputId = "cramerbuttonID",
                                    label = "Calculate Cramer's V"),
                         plotOutput(outputId = "fPlot", height = 700)
              ),
              tabPanel("Pivot tables",
                       sidebarPanel(
                         radioButtons(
                            inputId = "catradiobox1ID",
                            label = "First category",
                            choices = cat_names,
                            selected =  cat_names[1]),
                         width = 2),
                         sidebarPanel(
                           radioButtons(
                           inputId = "catradiobox2ID",
                           label = "Second category",
                           choices = cat_names,
                           selected = cat_names[2]),
                           width = 2),
                       mainPanel(
                         p(textOutput("count_pivot")),
                         tableOutput(outputId = "pivottableID")
                       )
              )
        )
    ),
    tabPanel("Manipulate data",
             tabsetPanel(
               tabPanel("RCode",
                        sidebarLayout(
                          sidebarPanel(
                            useShinyjs(), 
                            runcodeUI(code = Rscript, type = "textarea", height = 450),
                            actionButton("refreshID", "Refresh dataset")
                          ),
                          mainPanel(
                            textInput("nvarsID", "Number of variables"),
                            actionButton("runFS", "Run forward selection"),
                            p(textOutput("rmseFS")),
                            actionButton("runCV", "Run cross validation"),
                            p(textOutput("rmseCV"))
                          )
                        )
                        
               ),
               tabPanel("Categorical variables",
                        sidebarLayout(
                          sidebarPanel(                        
                            selectInput(
                              inputId = "selectCatID2",
                              label = "Select categorical variable",
                              choices = cat_names
                            ),
                            selectInput(
                              inputId = "selectTargetID2",
                              label = "Select numerical variable",
                              choices = all_num_names,
                              selected = target_name
                            ),
                            tableOutput(outputId = 'missingCatID2')
                          ),
                          mainPanel(
                            plotOutput(outputId = "catbarPlot2"),
                            plotOutput(outputId = "catboxPlot2")
                          )
                        )
               ),  
               tabPanel("Numerical variables",
                        sidebarLayout(
                          sidebarPanel(
                            selectInput(
                              inputId = "selectNumID2",
                              label = "Select numerical variable",
                              choices = num_names
                            ),
                            selectInput(
                              inputId = "selectNum2ID2",
                              label = "Select target",
                              choices = all_num_names,
                              selected = target_name
                            ),
                            tableOutput(outputId = 'missingNumID2'),
                            plotOutput(outputId = "corrPlot2")
                            
                          ),
                          mainPanel(
                            plotOutput(outputId = "numlinePlot2"),
                            plotOutput(outputId = "numPlot2")
                          )
                        )        
               ),
               tabPanel("Pivot tables",
                        sidebarPanel(
                          radioButtons(
                            inputId = "catradiobox1ID2",
                            label = "First category",
                            choices = cat_names,
                            selected =  cat_names[1]),
                          width = 2),
                        sidebarPanel(
                          radioButtons(
                            inputId = "catradiobox2ID2",
                            label = "Second category",
                            choices = cat_names,
                            selected = cat_names[2]),
                          width = 2),
                        mainPanel(p(textOutput("count_pivot2")),
                          tableOutput(outputId = "pivottableID2")
                          
                        )
               )
             )
    ),
    tabPanel("Run model",
            actionButton("runModel", "Run model and predict"),
            plotOutput(outputId = "results_plot")
    )
    
  )
)


server <- function(input, output, session) {
    
    runcodeServer()
  
    output$missingCatID <- renderTable({
        missing_cat <- cbind(column = cat_names, missing = sapply(df[,cat_names], function(x) sum(is.na(x))))
        missing_cat
    })
    
    output$missingNumID <- renderTable({
        missing_num <- cbind(column = c(num_names,target_name), missing = sapply(df_all_num, function(x) sum(is.na(x))))
        missing_num
    })
    
    output$catbarPlot <- renderPlot({ ggplot(data = df, aes_(as.name(input$selectCatID), fill = as.name(input$selectCatID))) + geom_bar() +  coord_flip()})
    
    output$catboxPlot <- renderPlot({ ggplot(df, aes_(as.name(input$selectCatID), as.name(input$selectTargetID), fill = as.name(input$selectCatID))) + geom_boxplot()+  coord_flip()})
    
    output$numlinePlot <- renderPlot({ggplot(data = df, aes_(as.name(input$selectNumID))) +  geom_density(colour = "red")})
    
    output$numPlot <- renderPlot({ggplot(data = df, aes_(as.name(input$selectNumID), as.name(input$selectNum2ID))) + geom_point(colour = "red")})
    
    output$corrPlot <- renderPlot({ ggcorrplot(corr_num, type = "upper",lab = TRUE, digits = 2)})
    
    
    
    paov <- eventReactive(input$FbuttonID, {
      n_comb <- n_cat*n_num2
      p_aov <- data.frame(matrix(rep(0, n_comb), ncol = n_num2))
      rownames(p_aov) <- cat_names
      colnames(p_aov) <- all_num_names
      c <- 1
      for (i in 1:n_cat) {
        cat.name <- cat_names[i]
        for (j in 1:n_num2) {
          num.name <- all_num_names[j]
          cc <- showNotification(paste0(cat.name, '&', num.name, " ", toString(c), "/", toString(n_comb)), duration = 25)
          myformula <- formula(paste(num.name, "~", cat.name, sep=""))
          my.aov <- summary(aov(myformula, df, na.action = na.omit))
          p_aov[i,j] <- round(my.aov[[1]][[5]][1], 4)
          c <- c + 1
        }
        
      }
      ggcorrplot(t(p_aov),lab = TRUE, digits = 3)
    })
    
    output$fPlot <- renderPlot({paov() })
    
    cramersV <- eventReactive(input$cramerbuttonID, {
      cramers <- run_cramersV(df)
      ggcorrplot(cramers, type = "lower",lab = TRUE, digits = 2)
    }
    )
    
    output$fPlot <- renderPlot({cramersV()})
    
    output$pivottableID <- renderTable({
      pivot_table <- df %>% 
        group_by_(input$catradiobox1ID, input$catradiobox2ID) %>% 
        summarize(count =  n(), mean_price_usd = mean(price_usd), min_price_usd = min(price_usd), max_price_usd = max(price_usd))
      
      pivot_table[order(pivot_table$mean_price_usd, decreasing = TRUE),]
    })
    
    output$count_pivot2 <- renderText({
      pivot_table <- df %>% group_by_(input$catradiobox1ID2, input$catradiobox2ID2) %>% summarize(count =  n())
      nrow(pivot_table)
    })
    
    
    

    ref_miss_cat <- eventReactive(input$refreshID,{
      missing_values(df_final, "cat")
    })
    
    output$missingCatID2 <- renderTable({
      missing_values(df_final, "cat")
    })
    
    output$missingCatID2 <- renderTable({
      ref_miss_cat()
    })
    
    ref_miss_num <- eventReactive(input$refreshID,{
      missing_values(df_final, "all_num")
    })
    
    output$missingNumID2 <- renderTable({
      missing_values(df_final, "all_num")
    })
    
    output$missingNumID2 <- renderTable({
      ref_miss_num()
    })
    
    output$catbarPlot2 <- renderPlot({
      ggplot(data = df_final, aes_(as.name(input$selectCatID2), fill = as.name(input$selectCatID2)))+ 
      geom_bar() +  coord_flip()})
    
    output$catboxPlot2 <- renderPlot({ 
      ggplot(df_final, aes_(as.name(input$selectCatID2), as.name(input$selectTargetID2), fill = as.name(input$selectCatID2))) +
      geom_boxplot()+  coord_flip()})
    
    output$numlinePlot2 <- renderPlot({ggplot(data = df_final, aes_(as.name(input$selectNumID2))) +  geom_density(colour = "red")})
    
    output$numPlot2 <- renderPlot({ggplot(data = df_final, aes_(as.name(input$selectNumID2), as.name(input$selectNum2ID2))) + geom_point(colour = "red")})
    
    output$corrPlot2 <- renderPlot({ ggcorrplot(cor(df[,get_col_names(df_final, "all_num")], use = "complete.obs"), type = "upper",lab = TRUE, digits = 2)})
    
    output$pivottableID2 <- renderTable({
      pivot_table <- df_final %>% 
        group_by_(input$catradiobox1ID2, input$catradiobox2ID2) %>%
        summarize(count =  n(), mean_price_usd = mean(price_usd), min_price_usd = min(price_usd), max_price_usd = max(price_usd))
      pivot_table[order(pivot_table$mean_price_usd, decreasing = TRUE),]
    })
    
    output$count_pivot2 <- renderText({
      pivot_table <- df_final %>% group_by_(input$catradiobox1ID2, input$catradiobox2ID2) %>% summarize(count =  n(), mean(price_usd))
      nrow(pivot_table)
    })
    
    observeEvent(input$refreshID,
                 updateSelectInput(
                   session,
                   inputId = "selectCatID2",
                   label = "Select categorical variable",
                   choices = (function(variables) if (length(variables)>0){ variables } else {'No categorical variable found'})(get_col_names(df_final, "cat"))
                 )
    )
    observeEvent(input$refreshID,
                 updateSelectInput(
                   session,
                   inputId = "selectTargetIDID2",
                   label = "Select target",
                   choices = get_col_names(df_final, "all_num"),
                   selected = target_name
                 )
    )
    observeEvent(input$refreshID,
                 updateSelectInput(
                   session,
                   inputId = "selectNumID2",
                   label = "Select numerical variable",
                   choices = get_col_names(df_final, "all_num")
                 )
    )
    observeEvent(input$refreshID,
                 updateSelectInput(
                   session,
                   inputId = "selectNum2ID2",
                   label = "Select target",
                   choices = get_col_names(df_final, "all_num"),
                   selected = target_name
                 )
    )
    observeEvent(input$refreshID,
                 updateRadioButtons(
                   session,
                   inputId = "catradiobox1ID2",
                   label = "First category",
                   choices = (function(variables) if (length(variables)>0){ variables } else {'No categorical variable found'})(get_col_names(df_final, "cat"))
                 )
    )
    observeEvent(input$refreshID,
                 updateRadioButtons(
                   session,
                   inputId = "catradiobox2ID2",
                   label = "Second category",
                   choices = (function(variables) if (length(variables)>0){ variables } else {'No categorical variable found'})(get_col_names(df_final, "cat"))
                 )
    )
    
    rmse_fs <- eventReactive(input$runFS,{
      results <- forward_sel(df_final, as.numeric(input$nvarsID))
      c(round(results$RMSE), results$vars)
    })
    
    output$rmseFS <- renderText({rmse_fs()})
    
    rmse_lm <- eventReactive(input$runCV,{CV_lm(df_final)})
    
    output$rmseCV <- renderText({rmse_lm()})
    
    output_plot <-eventReactive(input$runModel,{
      predictions <- Run_model(df, df_pred)
      melted_data <- melt(list(prices = df$price_usd, predictions = predictions$price_usd))
      return(ggplot(data = melted_data, aes(value, fill=L1)) +  geom_density())
      })

    output$results_plot <- renderPlot(output_plot())
    
}


shinyApp(ui = ui, server = server)
