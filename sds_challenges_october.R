library(shiny)
library(dplyr)
library(ggplot2)
library(ggcorrplot)
library(corrplot)
library(rcompanion)

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
corr_num <- cor(df_all_num)

n_comb <- n_cat*n_cat
cramers <- data.frame(matrix(rep(1, n_comb), ncol = n_cat))
rownames(cramers) <- cat_names
colnames(cramers) <- cat_names
c <- 1
for (i in 1:(n_cat-1)){
  cat.name <- cat_names[i]
  for (j in (i+1):n_cat){
    cat2.name <- cat_names[j]
    #cc <- showNotification(paste0(cat.name, '&', cat2.name, " ", toString(c), "/", toString(n_comb)), duration = 25)
    cramers[i,j] <- cramerV(df[,cat.name], df[,cat2.name])
    cramers[j,i] <- cramerV(df[,cat.name], df[,cat2.name])
  }
}

ui <- fluidPage(

    titlePanel("Car challenge"),
    
    navbarPage("My Application",
        tabPanel("Categorical variables",
            sidebarLayout(
                sidebarPanel(                        
                    selectInput(
                            inputId = "selectCatID",
                            label = "Select categorical variable",
                            choices = cat_names
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
                             label = "Select categorical variable",
                             choices = num_names
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
        tabPanel("Categoric relationships (Cramers V)",
                 plotOutput(outputId = "cramerID", height = 700)
        ),
        tabPanel("F-Statistic",
                   actionButton(inputId = "buttonID",
                                label = "Show F-Statistic"),
                   plotOutput(outputId = "fPlot", height = 700)
        ),
        tabPanel("Pivot tables",
                 sidebarPanel(
                   radioButtons(
                      inputId = "catradiobox1ID",
                      label = "First category",
                      choices = cat_names,
                      selected =  cat_names[5]),
                   width = 2),
                   sidebarPanel(
                     radioButtons(
                     inputId = "catradiobox2ID",
                     label = "Second category",
                     choices = cat_names,
                     selected = cat_names[7]),
                     width = 2),
                 mainPanel(
                   tableOutput(outputId = "pivottableID")
                 )
        )
    )
)


server <- function(input, output) {
    
    output$missingCatID <- renderTable({
        missing_cat <- cbind(column = cat_names, missing = sapply(df[,cat_names], function(x) sum(is.na(x))))
        missing_cat
    })
    
    output$missingNumID <- renderTable({
        missing_num <- cbind(column = c(num_names,target_name), missing = sapply(df_all_num, function(x) sum(is.na(x))))
        missing_num
    })
    
    output$catbarPlot <- renderPlot({ ggplot(data = df, aes_(as.name(input$selectCatID), fill = as.name(input$selectCatID))) + geom_bar()})
    
    output$catboxPlot <- renderPlot({ ggplot(df, aes_(as.name(input$selectCatID), as.name(target_name), fill = as.name(input$selectCatID))) + geom_boxplot()})
    
    output$numlinePlot <- renderPlot({ggplot(data = df, aes_(as.name(input$selectNumID))) +  geom_density(colour = "red")})
    
    output$numPlot <- renderPlot({ggplot(data = df, aes_(as.name(input$selectNumID), as.name(target_name))) + geom_point(colour = "red")})
    
    output$corrPlot <- renderPlot({ ggcorrplot(corr_num, type = "upper",lab = TRUE, digits = 2)})
    
    
    
    paov <- eventReactive(input$buttonID, {
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
          my.aov <- summary(aov(myformula, df, na.action=na.omit))
          p_aov[i,j] <- round(my.aov[[1]][[5]][1], 4)
          c <- c + 1
        }
        
      }
      ggcorrplot(t(p_aov),lab = TRUE, digits = 3)
    })
    
    output$fPlot <- renderPlot({paov() })
    
    output$pivottableID <- renderTable({
      pivot_table <- df %>% group_by_(input$catradiobox1ID, input$catradiobox2ID) %>% summarize(count =  n(), mean(price_usd))
      pivot_table
    })
    
    output$cramerID <- renderPlot({ggcorrplot(cramers, type = "lower",lab = TRUE, digits = 2)})
}


shinyApp(ui = ui, server = server)
