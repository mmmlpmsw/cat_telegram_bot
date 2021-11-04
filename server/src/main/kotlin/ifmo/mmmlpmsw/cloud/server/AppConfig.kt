package ifmo.mmmlpmsw.cloud.server

import org.springframework.context.annotation.ComponentScan
import org.springframework.context.annotation.Configuration
import org.springframework.web.servlet.config.annotation.EnableWebMvc

@Configuration
@EnableWebMvc
@ComponentScan(basePackages = ["ifmo.mmmlpmsw.cloud.server"])
open class AppConfig