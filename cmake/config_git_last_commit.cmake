find_program(GIT_EXECUTABLE NAMES git)

if (EXISTS ${GIT_EXECUTABLE})
    execute_process(
            COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
            RESULT_VARIABLE GIT_COMMIT_RESULT
            OUTPUT_VARIABLE GIT_COMMIT_ID
            OUTPUT_STRIP_TRAILING_WHITESPACE)

    if (GIT_COMMIT_RESULT EQUAL 0)
        add_compile_definitions(GIT_LAST_COMMIT=${GIT_COMMIT_ID})
        message(STATUS "set GIT_LAST_COMMIT to ${GIT_LAST_COMMIT} as compile definition")
    else()
        message(STATUS "Failed to git last commit with git")
    endif()

else()
    message(STATUS "Failed to find git command, not GIT_LAST_COMMIT will be set")
endif()
